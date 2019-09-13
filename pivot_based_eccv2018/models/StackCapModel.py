# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed = pack_padded_sequence(att_feats, list(att_masks.data.long().sum(1)), batch_first=True)
        return pad_packed_sequence(PackedSequence(module(packed[0]), packed[1]), batch_first=True)[0]
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.attri_feat_size = opt.attri_feat_size
        self.attri_hid_size = opt.attri_hid_size
        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential( nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.attri_embed = nn.Sequential(nn.Linear(self.attri_feat_size, self.input_encoding_size, bias=False),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _forward(self, fc_feats, attri_feats, att_feats, seq, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs0 = Variable(fc_feats.data.new(batch_size, seq.size(1) - 1, self.vocab_size + 1).zero_())
        outputs1 = Variable(fc_feats.data.new(batch_size, seq.size(1) - 1, self.vocab_size + 1).zero_())
        outputs2 = Variable(fc_feats.data.new(batch_size, seq.size(1) - 1, self.vocab_size + 1).zero_())

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        attri_feats = self.attri_embed(attri_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs2[:, i-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, fc_feats, attri_feats, att_feats, p_att_feats, state, att_masks)

            outputs0[:, i] = F.log_softmax(self.logit(output[0]))
            outputs1[:, i] = F.log_softmax(self.logit(output[1]))
            outputs2[:, i] = F.log_softmax(self.logit(output[2]))

        return [outputs0, outputs1, outputs2]

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_attri_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_attri_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
        logprobs = F.log_softmax(self.logit(output[-1]))

        return logprobs, state

    def _sample_beam(self, fc_feats, attri_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        attri_feats = self.attri_embed(attri_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_attri_feats = attri_feats[k:k + 1].expand(beam_size, attri_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(xt, tmp_fc_feats, tmp_attri_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
                logprobs = F.log_softmax(self.logit(output[-1]))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_attri_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return Variable(seq.transpose(0, 1)), Variable(seqLogprobs.transpose(0, 1))

    def _sample(self, fc_feats, attri_feats, att_feats, att_masks=None, opt={}):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, attri_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        attri_feats = self.attri_embed(attri_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        seq = Variable(fc_feats.data.new(batch_size, self.seq_length).long().zero_())
        seqLogprobs = Variable(fc_feats.data.new(batch_size, self.seq_length).zero_())
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                # seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            output, state = self.core(xt, fc_feats, attri_feats, att_feats, p_att_feats, state, att_masks)
            if decoding_constraint and t > 0:
                tmp = output[-1].data.new(output.size(0), self.vocab_size + 1).zero_()
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = F.log_softmax(self.logit(output[-1])+Variable(tmp))
            else:
                logprobs = F.log_softmax(self.logit(output[-1]))

        return seq, seqLogprobs

from .FCModel import LSTMCore
class StackCapCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackCapCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt_input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt_input_encoding_size + opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        #self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        #self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size*2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size*3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, attri_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([torch.add(xt, attri_feats),h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        #att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        att_res_2 = self.att2(h_1, att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([torch.add(xt, attri_feats),self.fusion1(torch.cat([h_0, h_1], 1)),att_res_2],1), [state[0][2:3], state[1][2:3]])

        return [h_0, h_1, self.fusion2(torch.cat([h_0, h_1, h_2], 1))], [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

'''
    input[1]: bz * xDim;
    input[2]: bz * oDim * yDim; output: bz * oDim;
    output: compute output scores: bz * L 
'''
class BilinearD3(nn.Module):
    def __init__(self, xDim, yDim, bias=True):
        super(BilinearD3, self).__init__()
        self.xDim = xDim
        self.yDim = yDim
        self.bias = bias or False
        self.x2y = nn.Linear(self.xDim, self.yDim, bias = self.bias)
        self.weight_init()

    def weight_init(self):
        init.xavier_uniform(self.x2y.weight, gain=np.sqrt(2))
        if self.bias:
            init.constant(self.x2y.bias, 0.1)

    '''
        # As = As.transpose(1, 2)  # input[2]: bz * oDim *yDim -->  bz * yDim *oDim
        #output = torch.bmm(map.view(-1, 1, self.yDim), As)  ## temp: (bz * 1 * xDim) * (bz * xDim * yDim) = bz * yDim
    '''
    def forward_bak(self, input, As):
        map = self.x2y(input) # bz * xDim --> bz * yDim
        output = Variable(torch.Tensor(As.size(0), As.size(1)).cuda(), requires_grad=True)
        for i in range(As.size(1)):
            output[:, i] = torch.sum(torch.bmm(map, As[:, i, :]), 1)

        return output

    def forward(self, input, As):
        map = self.x2y(input).unsqueeze(1) # bz * xDim --> bz * 1 * yDim
        As = As.transpose(1, 2)  # input[2]: bz * oDim *yDim -->  bz * yDim *oDim
        output = torch.bmm(map, As)  ## temp: (bz * 1 * xDim) * (bz * xDim * yDim) = bz * yDim

        return output.squeeze(1)

class Sentence_in_attention(nn.Module):
    def __init__(self, word_embed_size, m):
        super(Sentence_in_attention, self).__init__()
        self.m = m
        self.word_embed_size = word_embed_size

        self.BilinearD3 = BilinearD3(self.word_embed_size, self.word_embed_size, bias=False)
        self.l1 = nn.Linear(self.word_embed_size, self.m)
        self.weight_init()

    def weight_init(self):
        init.xavier_uniform(self.l1.weight, gain=np.sqrt(2))
        init.constant(self.l1.bias, 0.1)

    def makeWeightedSumUnit(self, x, alpha):
        g = torch.bmm(x.transpose(1,2), alpha.unsqueeze(2)) .squeeze(2)
        return g

        # prev_word_embedding: bz*512, As: bz*10*512
    def forward(self, prev_word_embed, As):
        attention_output = self.BilinearD3(prev_word_embed, As)  # bz*512
        beta = nn.Softmax()(attention_output)
        g_in = self.makeWeightedSumUnit(As, beta)  # bz*512
        temp = torch.add(g_in, prev_word_embed)
        output = self.l1(temp)
        return output

class Sentence_out_attention(nn.Module):
    def __init__(self, opt):
        super(Sentence_out_attention, self).__init__()
        self.drop_prob = opt.drop_prob_lm
        self.hDim = opt.rnn_size
        self.word_embed_size = opt.rnn_size
        self.weight = nn.Parameter(torch.rand(opt.rnn_size))
        self.BilinearD3 = BilinearD3(self.hDim, self.word_embed_size, bias=False)
        self.l1 = nn.Linear(self.word_embed_size, self.hDim)
        self.l2 = nn.Linear(self.hDim, self.hDim)
        self.weight_init()

    def weight_init(self):
        #init.xavier_uniform(self.l2.weight, gain=np.sqrt(2))
        init.xavier_uniform(self.l1.weight, gain=np.sqrt(2))
        init.constant(self.l1.bias, 0.1)
        init.xavier_uniform(self.l2.weight, gain=np.sqrt(2))
        init.constant(self.l2.bias, 0.1)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def makeWeightedSumUnit(self, x, alpha):
        g = torch.bmm(x.transpose(1,2), alpha.unsqueeze(2)) .squeeze(2)
        return g
        # h_t: bz*512, As: bz*10*512

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        attention_output = self.BilinearD3(h, F.tanh(att_feats.float()))
        beta = nn.Softmax()(attention_output)
        g_out = self.makeWeightedSumUnit(F.tanh(att_feats), beta)  # bz*512
        output = torch.add(self.l1(torch.mul(self.weight.expand(g_out.size(0), g_out.size(1)), g_out)), h)
        output = self.l2(nn.Dropout(self.drop_prob)(F.leaky_relu(output)))
        return output

# Attention Block for C_hat calculation
class Atten(nn.Module):
    def __init__(self, hidden_size):
        super(Atten, self).__init__()

        self.affine_v = nn.Linear(hidden_size, 49, bias=False)  # W_v
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)  # W_g
        self.affine_s = nn.Linear(hidden_size, 49, bias=False)  # W_s
        self.affine_h = nn.Linear(49, 1, bias=False)  # w_h

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)
        init.xavier_uniform(self.affine_s.weight)

    def forward(self, V, h_t, s_t):
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''

        # W_v * V + W_g * h_t * 1^T
        content_v = self.affine_v(self.dropout(V)).unsqueeze(1) \
                    + self.affine_g(self.dropout(h_t)).unsqueeze(2)

        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h(self.dropout(F.tanh(content_v))).squeeze(3)
        alpha_t = F.softmax(z_t.view(-1, z_t.size(2))).view(z_t.size(0), z_t.size(1), -1)

        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm(alpha_t, V).squeeze(2)

        # W_s * s_t + W_g * h_t
        content_s = self.affine_s(self.dropout(s_t)) + self.affine_g(self.dropout(h_t))
        # w_t * tanh( content_s )
        z_t_extended = self.affine_h(self.dropout(F.tanh(content_s)))

        # Attention score between sentinel and image content
        extended = torch.cat((z_t, z_t_extended), dim=2)
        alpha_hat_t = F.softmax(extended.view(-1, extended.size(2))).view(extended.size(0), extended.size(1), -1)
        beta_t = alpha_hat_t[:, :, -1]

        # c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        beta_t = beta_t.unsqueeze(2)
        c_hat_t = beta_t * s_t + (1 - beta_t) * c_t

        return c_hat_t, alpha_t, beta_t

# Sentinel BLock
class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(hidden_size, hidden_size, bias=False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Dropout applied before affine transformation
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform(self.affine_x.weight)
        init.xavier_uniform(self.affine_h.weight)

    def forward(self, x_t, h_t_1, cell_t):
        # g_t = sigmoid( W_x * x_t + W_h * h_(t-1) )
        gate_t = self.affine_x(self.dropout(x_t)) + self.affine_h(self.dropout(h_t_1[:,-1,:]))
        gate_t = F.sigmoid(gate_t)

        # Sentinel embedding
        s_t = gate_t * F.tanh(cell_t)

        return s_t

# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding
class AdaptiveBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(AdaptiveBlock, self).__init__()

        # Sentinel block
        self.sentinel = Sentinel(embed_size * 2, hidden_size)

        # Image Spatial Attention Block
        self.atten = Atten(hidden_size)

        # Final Caption generator
        # self.mlp = nn.Linear(hidden_size, vocab_size)

        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout(0.5)

        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        '''
        Initialize final classifier weights
        '''
        #init.kaiming_normal(self.mlp.weight, mode='fan_in')
        #self.mlp.bias.data.fill_(0)

    def forward(self, x, hiddens, cells, V):

        # hidden for sentinel should be h0-ht-1
        h0 = self.init_hidden(x.size(0))[0].transpose(0, 1)

        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        hiddens_t_1 = h0

        # Get Sentinel embedding, it's calculated blockly
        sentinel = self.sentinel(x, hiddens_t_1, cells)

        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights, beta = self.atten(V, hiddens, sentinel)

        # Final score along vocabulary
        scores = self.dropout(c_hat + hiddens)
        #scores = self.mlp(self.dropout(c_hat + hiddens))

        #return scores, atten_weights, beta
        return scores

    def init_hidden(self, bsz):
        '''
        Hidden_0 & Cell_0 initialization
        '''
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda()))
        else:
            return (Variable(weight.new(1, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(1, bsz, self.hidden_size).zero_()))

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class StackCapModel(AttModel):
    def __init__(self, opt):
        super(StackCapModel, self).__init__(opt)
        self.num_layers = 3
        self.core = StackCapCore(opt)