import os
import time
import logging
from itertools import chain, cycle
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import shutil
import models
import eval_utils
import misc.criterion as criterion
import misc.optimizer as optimizer
import models.NMT_Models as NMT_Models
import misc.utils as utils
from models.weight_init import *
from misc.rewards import init_scorer, get_self_critical_reward
from misc.nmt_translator import *

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None


def add_summary_value(writer, keys, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=keys, simple_value=value)])
    writer.add_summary(summary, iteration)


class Trainer(object):
    def __init__(self, opt, infos, loader, coco_loader):
        super(Trainer, self).__init__()
        self.opt = opt
        self.loader = loader
        self.coco_loader = coco_loader
        self.use_box_cls_prob = opt.use_box_cls_prob
        self.coco_eval_flag = opt.coco_eval_flag
        self.i2t_train_flag = opt.i2t_train_flag
        self.i2t_eval_flag = opt.i2t_eval_flag
        self.update_i2t_lr_flag = True
        self.best_flag_i2t = False
        self.tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)
        self.i2t_train_loss = 0.0
        self.i2t_avg_reward = 0.0
        self.i2t_val_loss = 0.0
        self.wemb_loss = 0.0
        self.lang_stats = {'Bleu_1': 0.0, 'Bleu_2': 0.0, 'Bleu_3': 0.0, 'Bleu_4': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0}
        self.best_i2t_val_score = infos.get('best_i2t_val_score', None) if opt.load_best_score == 1 else None
        self.optim = optimizer.Optim(opt)

        self.init_nmt(opt, infos)
        if opt.i2t_train_flag or opt.i2t_eval_flag: self.build_i2t(infos, loader)
        if opt.nmt_train_flag or opt.nmt_eval_flag: self.build_nmt(infos, loader, coco_loader)
        self.build_optimizer()

    def init_nmt(self, opt, infos):
        self.nmt_train_ppl = 0.0
        self.nmt_train_acc = 0.0
        self.nmt_valid_ppl = 0.0
        self.nmt_valid_acc = 0.0
        self.best_flag_nmt = False
        self.update_nmt_lr_flag = False
        self.nmt_eval_flag = opt.nmt_eval_flag
        self.nmt_train_flag = opt.nmt_train_flag
        self.best_nmt_val_acc = infos.get('best_nmt_val_acc', None) if opt.load_best_score == 1 else None

    def build_i2t(self, infos, loader):
        self.i2t_model = I2T_Model_init(self.opt, models.setup(self.opt))
        self.dp_i2t_model = torch.nn.DataParallel(self.i2t_model) if len(self.opt.gpus) > 1 else self.i2t_model
        self.dp_i2t_model.cuda()
        self.dp_i2t_model.training = True if self.i2t_train_flag else False
        self.i2t_crit = criterion.LanguageModelCriterion(self.opt)
        self.i2t_rl_crit = criterion.RewardCriterion()

    def build_nmt(self, infos, loader, coco_loader):
        print('>>  Create teacher NMT model')
        self.nmt_encoder = NMT_Models.Encoder(self.opt, loader.nmt_dicts['src'])
        self.nmt_decoder = NMT_Models.Decoder(self.opt, loader.nmt_dicts['tgt'])
        self.nmt_model = NMT_Models.NMTModel(self.opt, self.nmt_encoder, self.nmt_decoder, loader.nmt_dicts['src'], loader.nmt_dicts['tgt'], len(self.opt.gpus) > 1)
        self.nmt_generator = nn.Sequential(nn.Linear(self.opt.rnn_size, loader.nmt_dicts['tgt'].size()), nn.LogSoftmax())
        self.nmt_model, self.generator = NMT_Model_init(self.opt, self.nmt_model, self.nmt_generator)
        print('Initialize multi gpu: ... {}'.format(self.opt.gpus))
        self.dp_nmt_model = torch.nn.DataParallel(self.nmt_model, dim=1) if len(self.opt.gpus) > 1 else self.nmt_model
        self.dp_nmt_model.generator = torch.nn.DataParallel(self.nmt_generator, dim=0) if len(self.opt.gpus) > 1 else self.nmt_generator
        print('* number of parameters: %d' % sum([p.nelement() for p in self.dp_nmt_model.parameters()]))
        self.dp_nmt_model.cuda()
        self.dp_nmt_model.training = True if self.nmt_train_flag else False
        self.nmt_loss = criterion.NMTCriterion(loader.nmt_dicts['tgt'].size(), self.opt)
        self.nmt_crit = criterion.NMT_loss(self.opt, self.nmt_generator, self.nmt_loss)
        self.weight_trans = criterion.Weight_Trans(self.opt, loader)
        # self.Weight_Trans_y_= criterion.Weight_Trans_y(self.opt, loader, coco_loader)

    def save_models(self):
        print("Saving models to {}".format(self.opt.checkpoint_path))
        self.ckp_tag = '-best' if self.best_flag_i2t or self.best_flag_nmt else ''
        if self.nmt_train_flag: torch.save(self.dp_nmt_model.state_dict(), os.path.join(self.opt.checkpoint_path, 'model_nmt' + self.ckp_tag + '.pth'))
        if self.i2t_train_flag: torch.save(self.i2t_model.state_dict(), os.path.join(self.opt.checkpoint_path, 'model_i2t' + self.ckp_tag + '.pth'))
        if self.nmt_train_flag: torch.save(self.optim.nmt_optimizer, os.path.join(self.opt.checkpoint_path, 'optimizer_nmt' + self.ckp_tag + '.pth'))
        if self.i2t_train_flag: torch.save(self.optim.i2t_optimizer.state_dict(), os.path.join(self.opt.checkpoint_path, 'optimizer_i2t' + self.ckp_tag + '.pth'))

    def build_optimizer(self):
        self.optim.set_parameters(self.dp_i2t_model if self.i2t_train_flag else None, self.dp_nmt_model if self.nmt_train_flag else None)

    def encode_sequence(self, srcWords):
        srcData = self.loader.nmt_dicts['src'].convertToIdx(srcWords, onmt.Constants.UNK_WORD)
        return srcData.unsqueeze(0)

    def zh_en_mapping(self, zh_seq):
        start = time.time()
        en_seq = []
        srcBatch = []
        for i in range(len(zh_seq) / 5):
            srcTokens = self.encode_sequence(zh_seq[i * 5].split())
            srcBatch += [srcTokens]
        lengths = [srcBatch[i].shape[1] for i in range(len(srcBatch))]
        max_length = max(lengths)
        lengths = np.asarray(lengths)
        out = torch.Tensor(len(srcBatch), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(srcBatch)):
            out[i].narrow(0, 0, lengths[i]).copy_(srcBatch[i].view(1, -1))
        if out.dim() == 2:
            out = out.unsqueeze(2)
        indices = range(len(out))
        # within batch sorting by decreasing length for variable length rnns
        lengths, perm = torch.sort(torch.from_numpy(lengths.astype('int32')), 0, descending=True)
        indices = [indices[p] for p in perm]
        out = [out[p] for p in perm]
        lengths = lengths.view(1, -1)
        lengths = Variable(lengths, volatile=False)

        out = torch.stack(out, 0)
        out = out.transpose(0, 1).contiguous()

        return Variable(out, volatile=False).cuda(), lengths

    def train(self, data, loader, iteration, epoch, nmt_epoch):
        nmt_dec_state = None
        nmt_dec_state_zh = None
        torch.cuda.synchronize()
        self.optim.zero_grad()

        tmp = [data['fc_feats'], data['attri_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['nmt'] if self.nmt_train_flag else None]
        tmp = [_ if _ is None else (Variable(torch.from_numpy(_), requires_grad=False).cuda() if utils.under_0_4() else torch.from_numpy(_).cuda()) for _ in tmp]
        fc_feats, attri_feats, att_feats, labels, masks, att_masks, nmt_batch = tmp

        if self.i2t_train_flag:
            if self.update_i2t_lr_flag:
                self.optim.update_LearningRate('i2t', epoch)  # Assign the learning rate
                self.optim.update_ScheduledSampling_prob(self.opt, epoch, self.dp_i2t_model)  # Assign the scheduled sampling prob
                if self.opt.self_critical_after != -1 and epoch >= self.opt.self_critical_after:
                    # If start self critical training
                    self.sc_flag = True
                    init_scorer(self.opt.cached_tokens)
                else:
                    self.sc_flag = False
                self.update_i2t_lr_flag = False

            if not self.sc_flag:
                i2t_outputs = self.dp_i2t_model(fc_feats, attri_feats, att_feats, labels, att_masks)
                i2t_loss = self.i2t_crit(i2t_outputs, labels[:, 1:], masks[:, 1:])
            else:
                gen_result, sample_logprobs = self.dp_i2t_model(fc_feats, attri_feats, att_feats, att_masks, opt={'sample_max': 0}, mode='sample')
                reward = get_self_critical_reward(self.dp_i2t_model, fc_feats, attri_feats, att_feats, att_masks, data, gen_result, self.opt)
                i2t_loss = self.i2t_rl_crit(sample_logprobs, gen_result.data, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))

                self.i2t_avg_reward = np.mean(reward[:, 0])
            self.i2t_train_loss = i2t_loss.data[0] if utils.under_0_4() else i2t_loss.item()
            i2t_loss.backward(retain_graph=True)

        if self.nmt_train_flag:
            if self.update_nmt_lr_flag:
                self.optim.update_LearningRate('nmt', nmt_epoch)  # Assign the learning rate
            outputs, attn, dec_state, upper_bounds = self.dp_nmt_model(nmt_batch.src, nmt_batch.tgt, nmt_batch.lengths, nmt_dec_state)
            nmt_loss = self.nmt_crit(loader, nmt_batch, outputs, attn)

            if nmt_dec_state is not None: nmt_dec_state.detach()
            if nmt_dec_state_zh is not None: nmt_dec_state_zh.detach()

            self.nmt_crit.report_stats.n_src_words += nmt_batch.lengths.data.sum()
            self.nmt_train_ppl = self.nmt_crit.report_stats.ppl()
            self.nmt_train_acc = self.nmt_crit.report_stats.accuracy()
            # Minimize the word embedding weights
            # wemb_weight_loss = self.weight_trans(self.i2t_model.embed, self.nmt_encoder.embeddings.word_lut)
            # self.wemb_loss = wemb_weight_loss.data[0]

            nmt_loss.backward(retain_graph=True)
        # if self.nmt_train_flag: wemb_weight_loss.backward(retain_graph=True)
        self.optim.step()

    def eval(self, loader, coco_loader):
        self.predictions = []
        # make evaluation on validation set, and save model
        eval_kwargs = {'split': 'val', 'dataset': self.opt.input_json}
        eval_kwargs.update(vars(self.opt))
        print(eval_kwargs)
        eval_tmp = eval_utils.eval_split(self.opt, loader, self.dp_i2t_model if self.i2t_eval_flag else None, self.dp_nmt_model if self.nmt_eval_flag else None, eval_kwargs)
        if self.coco_eval_flag:
            self.i2t_val_loss, self.predictions, self.coco_predictions, self.lang_stats, self.coco_lang_stats, self.nmt_valid_ppl, self.nmt_valid_acc = eval_tmp
        if self.i2t_eval_flag:
            self.i2t_val_loss, self.predictions, self.lang_stats, self.nmt_valid_ppl, self.nmt_valid_acc = eval_tmp
            self.coco_lang_stats = self.lang_stats
            current_score = self.lang_stats['CIDEr'] if self.opt.language_eval == 1 else - self.i2t_val_loss
            if self.best_i2t_val_score is None or current_score > self.best_i2t_val_score:
                self.best_i2t_val_score = current_score
                self.best_flag_i2t = True
        # Save model if is improving on validation result
        current_nmt_val_acc = self.nmt_valid_acc
        if self.nmt_eval_flag and (self.best_nmt_val_acc is None or current_nmt_val_acc > self.best_nmt_val_acc):
            self.best_nmt_val_acc = current_nmt_val_acc
            self.best_flag_nmt = True