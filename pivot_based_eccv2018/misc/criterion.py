from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import time
import math
import sys
import misc.utils as utils
import misc.constants as constants

def shardVariables(variables, batches, eval):
    """
    Split a dict of variables up into sharded dummy
    variables.
    """
    dummies = {}
    n_shards = ((list(variables.values())[0].size(0) - 1) // batches) + 1
    shards = [{} for _ in range(n_shards)]
    for k in variables:
        if isinstance(variables[k], Variable) and variables[k].requires_grad:
            dummies[k] = Variable(variables[k].data, requires_grad=(not eval),
                                  volatile=eval)
        else:
            dummies[k] = variables[k]
        splits = torch.split(dummies[k], batches)
        for i, v in enumerate(splits):
            shards[i][k] = v
    return shards, dummies

def collectGrads(variables, dummy):
    """Given a set of variables, find the ones with gradients"""
    inputs = []
    grads = []
    for k in dummy:
        if isinstance(variables[k], Variable) and (dummy[k].grad is not None):
            inputs.append(variables[k])
            grads.append(dummy[k].grad.data)
    return inputs, grads

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        '''
        input = utils.to_contiguous(input).view(-1)
        reward = utils.to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = utils.to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = (- input.float() * reward.float() * Variable(mask)) if utils.under_0_4() else (- input.float() * reward.float() * mask)
        output = torch.sum(output.float()) / torch.sum(mask.float())
        '''
        input = utils.to_contiguous(input).view(-1)
        reward = utils.to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = utils.to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output

def NMTCriterion(vocabSize, opt):
    """
    Construct the standard NMT Criterion
    """
    weight = torch.ones(vocabSize)
    weight[constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    #crit = nn.NLLLoss(size_average=False, ignore_index=onmt.Constants.PAD)
    if opt.gpus:
        crit.cuda()
    return crit

class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()
        self.caption_model = opt.caption_model

    def xe_loss(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

    def forward(self, input, target, mask):
        if 'stackcap' in self.caption_model:
            output = self.xe_loss(input[0], target, mask) + self.xe_loss(input[1], target, mask) + self.xe_loss(input[2], target, mask)
            #output = self.xe_loss(input[-1], target, mask)
        else:
            output = self.xe_loss(input, target, mask)

        return output

class NMT_loss(nn.Module):
    def __init__(self, opt, generator, crit, eval=False):
        super(NMT_loss, self).__init__()
        self.generator = generator
        self.crit = crit
        self.batch_size = opt.batch_size
        self.copy_loss = opt.copy_attn,
        self.lambda_coverage = opt.lambda_coverage
        self.lambda_fertility = opt.lambda_fertility
        self.lambda_exhaust = opt.lambda_exhaust
        self.mse = torch.nn.MSELoss()
        self.total_stats = Statistics()
        self.report_stats = Statistics()

    def score(self, loss_t, scores_t, targ_t):
        pred_t = scores_t.data.max(1)[1]
        non_padding = targ_t.ne(constants.PAD).data
        num_correct_t = pred_t.eq(targ_t.data.long()).masked_select(non_padding).sum()
        return Statistics(loss_t.data[0], non_padding.sum(), num_correct_t)

    def compute_std_loss(self, out_t, targ_t):
        scores_t = self.generator(out_t)
        loss_t = self.crit(scores_t, targ_t.view(-1).long())
        return loss_t, scores_t

    def forward(self, loader, batch, outputs, attns):
        nmt_stats_reset = (loader.nmt_batchIdx > len(loader.nmt_trainData)) or False
        stats = Statistics()

        original = {"out_t": outputs, "targ_t": batch.tgt[1:]}
        #original["align_t"] = batch.alignment[1:]

        def bottle(v):
            return v.view(-1, v.size(2))

        loss_t, scores_t = self.compute_std_loss(bottle(original["out_t"]), original["targ_t"])
        stats.update(self.score(loss_t, scores_t, original["targ_t"]))

        self.total_stats.update(stats)
        self.report_stats.update(stats)
        if nmt_stats_reset:
            self.total_stats = Statistics()
            self.report_stats = Statistics()

        return loss_t

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]


class KLD(nn.Module):
    def __init__(self, opt, generator):
        super(KLD, self).__init__()
        self.kl_loss = nn.KLDivLoss()
        self.generator = generator
    def forward(self, y0, y1):
        loss = self.kl_loss(self.generator(y0), self.generator(y1))
        return loss

class Weight_Trans(nn.Module):
    def __init__(self, opt, loader):
        super(Weight_Trans, self).__init__()
        self.nmt_pivot_idx2label = loader.nmt_dicts['src'].idxToLabel
        self.nmt_pivot_label2idx = loader.nmt_dicts['src'].labelToIdx
        self.i2t_pivot_idx2label = loader.ix_to_word
        self.i2t_pivot_label2idx = {label: idx for idx, label in self.i2t_pivot_idx2label.items()}
        self.i2t_pivot_joint_mask = []
        self.nmt_pivot_joint_mask = []
        self.i2t_pivot_joint_vocab = {}
        self.nmt_pivot_joint_vocab = {}
        self.mse = torch.nn.MSELoss()
        self.l2 = L2Loss(opt)
        self.gen_joint_mask(opt)
        path = 'save/20180222-093200.fc/model-best.pth'
        if os.path.isfile(path):
            other = torch.load()
            self.i2t_wemb_weights = other.items()[6][1].cpu()

    def gen_joint_mask(self, opt):
        print('Generate mask for joint vocabulary.')
        head, tail = os.path.split(opt.input_nmt_pt)
        pre_head = opt.input_nmt_pt.replace('.train.pt', '')
        if os.path.exists(pre_head +'pivot.joint_vocab.pt'):
            load_data = torch.load(pre_head + 'pivot.joint_vocab.pt')
            self.i2t_pivot_joint_vocab = load_data['i2t_pivot_joint_vocab']
            self.nmt_pivot_joint_vocab = load_data['nmt_pivot_joint_vocab']
            self.i2t_pivot_joint_mask = load_data['i2t_pivot_joint_mask']
            self.nmt_pivot_joint_mask = load_data['nmt_pivot_joint_mask']
        else:
            for idx, label in self.i2t_pivot_idx2label.items():
                if self.i2t_pivot_idx2label[idx] in self.nmt_pivot_label2idx.keys():
                    #print(label.encode('utf-8'))
                    self.i2t_pivot_joint_mask.append(1)
                    self.i2t_pivot_joint_vocab[idx] = label
                else:
                    self.i2t_pivot_joint_mask.append(0)
            for idx, label in self.nmt_pivot_idx2label.items():
                if self.nmt_pivot_idx2label[idx] in self.i2t_pivot_label2idx.keys():
                    #print(label.encode('utf-8'))
                    self.nmt_pivot_joint_mask.append(1)
                    self.nmt_pivot_joint_vocab[idx] = label
                else:
                    self.nmt_pivot_joint_mask.append(0)

            save_data = {'i2t_pivot_joint_vocab': self.i2t_pivot_joint_vocab,
                         'nmt_pivot_joint_vocab': self.nmt_pivot_joint_vocab,
                         'i2t_pivot_joint_mask': self.i2t_pivot_joint_mask,
                         'nmt_pivot_joint_mask': self.nmt_pivot_joint_mask}
            torch.save(save_data, opt.input_nmt_pt.replace('.train.pt', '')+'pivot.joint_vocab.pt')

        self.maps = torch.Tensor(len(self.i2t_pivot_joint_vocab), 2)
        nmt_pivot_joint_vocab_label2idx = {label: idx for idx, label in self.nmt_pivot_joint_vocab.items()}
        vocab_idx = 0
        for idx, label in self.i2t_pivot_joint_vocab.items():
            if label in nmt_pivot_joint_vocab_label2idx.keys():
                self.maps[vocab_idx,0] = int(idx)
                self.maps[vocab_idx,1] = nmt_pivot_joint_vocab_label2idx[label]
                vocab_idx = vocab_idx + 1
        print('Joint vocabulary for i2t = {} and t2i = {}'.format(len(self.i2t_pivot_joint_vocab), len(self.nmt_pivot_joint_vocab)))

    def mse_loss(self, input, target):
        #return torch.sum((input - target)^2) / input.data.nelement()
        return torch.mean((input - target) ** 2)

    def forward(self, pivot_wemb_i2t, pivot_wemb_nmt):
        #_pivot_wemb_i2t = self.i2t_wemb_weights[self.maps[:, 0].long()]
        _pivot_wemb_i2t = pivot_wemb_i2t(Variable(torch.from_numpy(self.maps[:, 0].long().numpy()).cuda(), requires_grad= False))
        _pivot_wemb_nmt = pivot_wemb_nmt(Variable(torch.from_numpy(self.maps[:, 1].long().numpy()).cuda(), requires_grad= False))
        loss_0 = self.mse_loss(_pivot_wemb_nmt, _pivot_wemb_i2t)
        return loss_0

class Weight_Trans_y(nn.Module):
    def __init__(self, opt, loader, coco_loader):
        super(Weight_Trans_y, self).__init__()
        self.nmt_target_idx2label = loader.nmt_dicts['tgt'].idxToLabel
        self.nmt_target_label2idx = loader.nmt_dicts['tgt'].labelToIdx
        self.i2t_target_idx2label = coco_loader.ix_to_word
        self.i2t_target_label2idx = {label: idx for idx, label in self.i2t_target_idx2label.items()}
        self.i2t_target_joint_mask = []
        self.nmt_target_joint_mask = []
        self.i2t_target_joint_vocab = {}
        self.nmt_target_joint_vocab = {}
        self.mse = torch.nn.MSELoss()
        self.l2 = L2Loss(opt)
        self.gen_joint_mask(opt)
        other = torch.load(open('save/09021117_cnn_resnet101.lm_debug13_scst_.rnn_LSTM/09021117_cnn_resnet101.lm_debug13_scst_.rnn_LSTM.model-best.pth'))
        self.i2t_wemb_weights = other.items()[6][1].cpu()

    def gen_joint_mask(self, opt):
        print('Generate mask for joint vocabulary.')
        head, tail = os.path.split(opt.input_nmt_pt)
        pre_head = opt.input_nmt_pt.replace('.train.pt', '')
        if os.path.exists(pre_head +'target.joint_vocab.pt'):
            load_data = torch.load(pre_head + 'target.joint_vocab.pt')
            self.i2t_target_joint_vocab = load_data['i2t_target_joint_vocab']
            self.nmt_target_joint_vocab = load_data['nmt_target_joint_vocab']
            self.i2t_target_joint_mask = load_data['i2t_target_joint_mask']
            self.nmt_target_joint_mask = load_data['nmt_target_joint_mask']
        else:
            for idx, label in self.i2t_target_idx2label.items():
                if self.i2t_target_idx2label[idx] in self.nmt_target_label2idx.keys():
                    #print(label.encode('utf-8'))
                    self.i2t_target_joint_mask.append(1)
                    self.i2t_target_joint_vocab[idx] = label
                else:
                    self.i2t_target_joint_mask.append(0)
            for idx, label in self.nmt_target_idx2label.items():
                if self.nmt_target_idx2label[idx] in self.i2t_target_label2idx.keys():
                    #print(label.encode('utf-8'))
                    self.nmt_target_joint_mask.append(1)
                    self.nmt_target_joint_vocab[idx] = label
                else:
                    self.nmt_target_joint_mask.append(0)

            save_data = {'i2t_target_joint_vocab': self.i2t_target_joint_vocab,
                         'nmt_target_joint_vocab': self.nmt_target_joint_vocab,
                         'i2t_target_joint_mask': self.i2t_target_joint_mask,
                         'nmt_target_joint_mask': self.nmt_target_joint_mask}
            torch.save(save_data, opt.input_nmt_pt.replace('.train.pt', '')+'target.joint_vocab.pt')

        self.maps = torch.Tensor(len(self.i2t_target_joint_vocab), 2)
        nmt_target_joint_vocab_label2idx = {label: idx for idx, label in self.nmt_target_joint_vocab.items()}
        vocab_idx = 0
        for idx, label in self.i2t_target_joint_vocab.items():
            if label in nmt_target_joint_vocab_label2idx.keys():
                self.maps[vocab_idx,0] = int(idx)
                self.maps[vocab_idx,1] = nmt_target_joint_vocab_label2idx[label]
                vocab_idx = vocab_idx + 1
        print('Joint vocabulary for i2t = {} and t2i = {}'.format(len(self.i2t_target_joint_vocab), len(self.nmt_target_joint_vocab)))

    def mse_loss(self, input, target):
        #return torch.sum((input - target)^2) / input.data.nelement()
        return torch.mean((input - target) ** 2)

    def forward(self, pivot_wemb_i2t, pivot_wemb_nmt):
        #_pivot_wemb_i2t = self.i2t_wemb_weights[self.maps[:, 0].long()]
        _pivot_wemb_i2t = pivot_wemb_i2t(Variable(torch.from_numpy(self.maps[:, 0].long().numpy()).cuda(), requires_grad= False))
        _pivot_wemb_nmt = pivot_wemb_nmt(Variable(torch.from_numpy(self.maps[:, 1].long().numpy()).cuda(), requires_grad= False))
        loss_0 = self.mse_loss(_pivot_wemb_nmt, _pivot_wemb_i2t)
        return loss_0