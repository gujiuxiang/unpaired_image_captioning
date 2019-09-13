from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import misc.utils as utils
import torch

def freeze_param(model):
    for param in model.parameters():
        param.requires_grad = False

def activate_param(model):
    for param in model.parameters():
        param.requires_grad = True


def I2T_Model_init_layer(model, other):
    for i in range(len(other.items())):
        if 'embed.weight' in other.items()[i][0]:
            print("> Initialize embed")
            del model.embed.weight
            model.embed.weight = nn.Parameter(other.items()[i][1])
        if 'img_embed.weight' in other.items()[i][0]:
            print("> Initialize img_embed")
            del model.img_embed.weight
            del model.img_embed.bias
            model.img_embed.weight = nn.Parameter(other.items()[i][1])
            model.img_embed.bias = nn.Parameter(other.items()[i + 1][1])
            # freeze_param(model.img_embed)
        if 'core.i2h.weight' in other.items()[i][0]:
            print("> Initialize core")
            del model.core.i2h.weight
            del model.core.i2h.bias
            del model.core.h2h.weight
            del model.core.h2h.bias
            model.core.i2h.weight = nn.Parameter(other.items()[i][1])
            model.core.i2h.bias = nn.Parameter(other.items()[i + 1][1])
            model.core.h2h.weight = nn.Parameter(other.items()[i + 2][1])
            model.core.h2h.bias = nn.Parameter(other.items()[i + 3][1])
            # freeze_param(model.core)
        if 'logit.weight' in other.items()[i][0]:
            print("> Initialize logit")
            del model.logit.weight
            del model.logit.bias
            model.logit.weight = nn.Parameter(other.items()[i][1])
            model.logit.bias = nn.Parameter(other.items()[i + 1][1])
    return model

def I2T_Model_init(opt, model):
    if os.path.isfile(os.path.join(opt.start_from, "model_i2t-best.pth")):
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from, "infos-best.pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
        print("Load pretrained i2t model from: {}".format(os.path.join(opt.start_from, 'model_i2t-best.pth')))
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model_i2t-best.pth')))
        #other = torch.load(os.path.join(opt.start_from, 'model_i2t-best.pth'))
        #model = I2T_Model_init_layer(model, other)
    else:
        # Here we only init the fc model (we use it as the baseline)
        path = ''
        if len(path)>0:
            other = torch.load(path)
            for i in range(len(other.items())):
                if 'embed.weight' in other.items()[i][0]:
                    print("  > Initialize embed")
                    del model.embed.weight
                    model.embed.weight = nn.Parameter(other.items()[i][1])
                if 'img_embed.weight' in other.items()[i][0]:
                    print("  > Initialize img_embed")
                    del model.img_embed.weight
                    del model.img_embed.bias
                    model.img_embed.weight = nn.Parameter(other.items()[i][1])
                    model.img_embed.bias = nn.Parameter(other.items()[i + 1][1])
                    #freeze_param(model.img_embed)
                if 'core.i2h.weight' in other.items()[i][0]:
                    print("  > Initialize core")
                    del model.core.i2h.weight
                    del model.core.i2h.bias
                    del model.core.h2h.weight
                    del model.core.h2h.bias
                    model.core.i2h.weight = nn.Parameter(other.items()[i][1])
                    model.core.i2h.bias = nn.Parameter(other.items()[i + 1][1])
                    model.core.h2h.weight = nn.Parameter(other.items()[i + 2][1])
                    model.core.h2h.bias = nn.Parameter(other.items()[i + 3][1])
                    #freeze_param(model.core)
                if 'logit.weight' in other.items()[i][0]:
                    print("  > Initialize logit")
                    del model.logit.weight
                    del model.logit.bias
                    model.logit.weight = nn.Parameter(other.items()[i][1])
                    model.logit.bias = nn.Parameter(other.items()[i + 1][1])
    return model


def NMT_Model_init_(opt, model, generator):
    init_path = 'save/20180228-011203.fcnmt_True/model_nmt-best.pth'
    print('Initialize model with pretrained model {}'.format(init_path))
    other = torch.load(init_path)
    #model_state_dict = other['model']
    #generator_state_dict = other['generator']
    #model.load_state_dict(model_state_dict)
    model.load_state_dict(other)
    #generator.load_state_dict(generator_state_dict)
    #return model, generator
    return model

def NMT_Model_init_layer(model, generator, other):
    for i in range(len(other.items())):
        if 'encoder.rnn.bias_hh_l0_reverse' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.bias_hh_l0_reverse')
            del model.encoder.rnn.bias_hh_l0_reverse
            model.encoder.rnn.bias_hh_l0_reverse = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.bias_hh_l0_reverse' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.bias_hh_l0_reverse')
            del model.encoder.rnn.bias_hh_l0_reverse
            model.encoder.rnn.bias_hh_l0_reverse = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.weight_ih_l0_reverse' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.weight_ih_l0_reverse')
            del model.encoder.rnn.weight_ih_l0_reverse
            model.encoder.rnn.weight_ih_l0_reverse = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.weight_hh_l0' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.weight_hh_l0')
            del model.encoder.rnn.weight_hh_l0
            model.encoder.rnn.weight_hh_l0 = nn.Parameter(other.items()[i][1])
        if 'decoder.attn.linear_in.weight' in other.items()[i][0]:
            print('  > Initialize decoder.attn.linear_in.weight')
            del model.decoder.attn.linear_in.weight
            model.decoder.attn.linear_in.weight = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.weight_ih_l0' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.weight_ih_l0')
            del model.encoder.rnn.weight_ih_l0
            model.encoder.rnn.weight_ih_l0 = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.bias_ih_l0' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.bias_ih_l0')
            del model.encoder.rnn.bias_ih_l0
            model.encoder.rnn.bias_ih_l0 = nn.Parameter(other.items()[i][1])
        if 'decoder.rnn.layers.0.bias_ih' in other.items()[i][0]:
            print('  > Initialize decoder.rnn.layers.0.bias_ih')
            del model.decoder.rnn.layers[0].bias_ih
            model.decoder.rnn.layers[0].bias_ih = nn.Parameter(other.items()[i][1])
        if 'encoder.embeddings.word_lut.weight' in other.items()[i][0]:
            print('  > Initialize encoder.embeddings.word_lut.weight')
            del model.encoder.embeddings.word_lut.weight
            model.encoder.embeddings.word_lut.weight = nn.Parameter(other.items()[i][1])
        if 'decoder.rnn.layers.0.bias_hh' in other.items()[i][0]:
            print('  > Initialize decoder.rnn.layers.0.bias_hh')
            del model.decoder.rnn.layers[0].bias_hh
            model.decoder.rnn.layers[0].bias_hh = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.bias_hh_l0' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.bias_hh_l0')
            del model.encoder.rnn.bias_hh_l0
            model.encoder.rnn.bias_hh_l0 = nn.Parameter(other.items()[i][1])
        if 'encoder.embeddings.linear.bias' in other.items()[i][0]:
            print('  > Initialize encoder.embeddings.linear.bias')
            del model.encoder.embeddings.linear.bias
            model.encoder.embeddings.linear.bias = nn.Parameter(other.items()[i][1])
        if 'decoder.rnn.layers.0.weight_hh' in other.items()[i][0]:
            print('  > Initialize decoder.rnn.layers.0.weight_hh')
            del model.decoder.rnn.layers[0].weight_hh
            model.decoder.rnn.layers[0].weight_hh = nn.Parameter(other.items()[i][1])
        if 'decoder.rnn.layers.0.weight_ih' in other.items()[i][0]:
            print('  > Initialize decoder.rnn.layers.0.weight_ih')
            del model.decoder.rnn.layers[0].weight_ih
            model.decoder.rnn.layers[0].weight_ih = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.bias_ih_l0_reverse' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.bias_ih_l0_reverse')
            del model.encoder.rnn.bias_ih_l0_reverse
            model.encoder.rnn.bias_ih_l0_reverse = nn.Parameter(other.items()[i][1])
        if 'decoder.embeddings.word_lut.weight' in other.items()[i][0]:
            print('  > Initialize decoder.embeddings.word_lut.weight')
            del model.decoder.embeddings.word_lut.weight
            model.decoder.embeddings.word_lut.weight = nn.Parameter(other.items()[i][1])
        if 'encoder.embeddings.linear.weight' in other.items()[i][0]:
            print('  > Initialize encoder.embeddings.linear.weight')
            del model.encoder.embeddings.linear.weight
            model.encoder.embeddings.linear.weight = nn.Parameter(other.items()[i][1])
        if 'encoder.rnn.weight_hh_l0_reverse' in other.items()[i][0]:
            print('  > Initialize encoder.rnn.weight_hh_l0_reverse')
            del model.encoder.rnn.weight_hh_l0_reverse
            model.encoder.rnn.weight_hh_l0_reverse = nn.Parameter(other.items()[i][1])
        if 'decoder.attn.linear_out.weight' in other.items()[i][0]:
            print('  > Initialize decoder.attn.linear_out.weight')
            del model.decoder.attn.linear_out.weight
            model.decoder.attn.linear_out.weight = nn.Parameter(other.items()[i][1])

        if 'generator.0.weight' in other.items()[i][0]:
            print('  > Initialize generator.0.weight')
            del generator[0].weight
            del generator[0].bias
            generator[0].weight = nn.Parameter(other.items()[i][1])
            generator[0].bias = nn.Parameter(other.items()[i + 1][1])
    return model, generator

def NMT_Model_init(opt, model, generator):
    if os.path.isfile(os.path.join(opt.start_from, "model_nmt-best.pth")):
        print('Loading NMT dicts from checkpoint at %s' % os.path.join(opt.start_from, "model_nmt-best.pth"))
        other = torch.load(os.path.join(opt.start_from, "model_nmt-best.pth"), map_location=lambda storage, loc: storage)
        print('Loading model from checkpoint at %s' % opt.start_from)
        #model, generator = NMT_Model_init_layer(model, generator, other)
        model.generator = generator
        model.load_state_dict(other)
        return model, model.generator
        #return model, generator
    else:
        init_choice = 1
        if init_choice == 0:
            init_path = 'save/nmt/demo-model-0303-full_acc_54.98_ppl_8.91_e15.pt'
            print('  > Initialize model with pretrained model {}'.format(init_path))
            other = torch.load(init_path)

            del model.encoder.rnn.bias_hh_l0_reverse
            del model.encoder.rnn.weight_ih_l0_reverse
            del model.encoder.rnn.weight_hh_l0
            del model.decoder.attn.linear_in.weight
            del model.encoder.rnn.weight_ih_l0
            del model.encoder.rnn.bias_ih_l0
            del model.decoder.rnn.layers[0].bias_ih
            del model.encoder.embeddings.word_lut.weight
            del model.decoder.rnn.layers[0].bias_hh
            del model.encoder.rnn.bias_hh_l0
            del model.encoder.embeddings.linear.bias
            del model.decoder.rnn.layers[0].weight_hh
            del model.decoder.rnn.layers[0].weight_ih
            del model.encoder.rnn.bias_ih_l0_reverse
            del model.decoder.embeddings.word_lut.weight
            del model.encoder.embeddings.linear.weight
            del model.encoder.rnn.weight_hh_l0_reverse
            del model.decoder.attn.linear_out.weight

            model.encoder.rnn.bias_hh_l0_reverse        = nn.Parameter(other.items()[5][1].items()[0][1])
            model.encoder.rnn.weight_ih_l0_reverse      = nn.Parameter(other.items()[5][1].items()[1][1])
            model.encoder.rnn.weight_hh_l0              = nn.Parameter(other.items()[5][1].items()[2][1])
            model.decoder.attn.linear_in.weight         = nn.Parameter(other.items()[5][1].items()[3][1])
            model.encoder.rnn.weight_ih_l0              = nn.Parameter(other.items()[5][1].items()[4][1])
            model.encoder.rnn.bias_ih_l0                = nn.Parameter(other.items()[5][1].items()[5][1])
            model.decoder.rnn.layers[0].bias_ih         = nn.Parameter(other.items()[5][1].items()[6][1])
            model.encoder.embeddings.word_lut.weight    = nn.Parameter(other.items()[5][1].items()[7][1])
            model.decoder.rnn.layers[0].bias_hh         = nn.Parameter(other.items()[5][1].items()[8][1])
            model.encoder.rnn.bias_hh_l0                = nn.Parameter(other.items()[5][1].items()[9][1])
            model.encoder.embeddings.linear.bias        = nn.Parameter(other.items()[5][1].items()[10][1])
            model.decoder.rnn.layers[0].weight_hh       = nn.Parameter(other.items()[5][1].items()[11][1])
            model.decoder.rnn.layers[0].weight_ih       = nn.Parameter(other.items()[5][1].items()[12][1])
            model.encoder.rnn.bias_ih_l0_reverse        = nn.Parameter(other.items()[5][1].items()[13][1])
            model.decoder.embeddings.word_lut.weight    = nn.Parameter(other.items()[5][1].items()[14][1])
            model.encoder.embeddings.linear.weight      = nn.Parameter(other.items()[5][1].items()[15][1])
            model.encoder.rnn.weight_hh_l0_reverse      = nn.Parameter(other.items()[5][1].items()[16][1])
            model.decoder.attn.linear_out.weight        = nn.Parameter(other.items()[5][1].items()[17][1])

            del generator[0].weight
            del generator[0].bias
            generator[0].weight = nn.Parameter(other.items()[1][1].items()[0][1])
            generator[0].bias   = nn.Parameter(other.items()[1][1].items()[1][1])
        else:
            init_path = 'save/20180228-011203.fcnmt_True/model_nmt-best.pth'
            if os.path.isfile(init_path):
                print('Initialize model with pretrained model {}'.format(init_path))
                other = torch.load(init_path)
                model, generator = NMT_Model_init_layer(model, generator, other)
        return model, generator

def StackCapModel_init(model):
    '''
    other = torch.load('save/20171221-113000.stackcap/model-best.pth')
    for i in range(len(other.items())):
        if 'embed.0.weight' in other.items()[i][0]:
            print("Initialize embed")
            del model.embed[0].weight
            model.embed[0].weight = nn.Parameter(other.items()[i][1])
        if 'fc_embed.0.weight' in other.items()[i][0]:
            print("Initialize fc_embed")
            del model.fc_embed[0].weight
            del model.fc_embed[0].bias
            model.fc_embed[0].weight = nn.Parameter(other.items()[i][1])
            model.fc_embed[0].bias = nn.Parameter(other.items()[i+1][1])
        if 'att_embed.0.weight' in other.items()[i][0]:
            print("Initialize att_embed")
            del model.att_embed[1].weight
            del model.att_embed[1].bias
            model.att_embed[1].weight = nn.Parameter(other.items()[i][1])
            model.att_embed[1].bias = nn.Parameter(other.items()[i+1][1])
        if 'logit.weight' in other.items()[i][0]:
            print("Initialize logit")
            del model.logit.weight
            del model.logit.bias
            model.logit.weight = nn.Parameter(other.items()[i][1])
            model.logit.bias = nn.Parameter(other.items()[i+1][1])
        if 'ctx2att.weight' in other.items()[i][0]:
            print("Initialize ctx2att")
            del model.ctx2att.weight
            del model.ctx2att.bias
            model.ctx2att.weight = nn.Parameter(other.items()[i][1])
            model.ctx2att.bias = nn.Parameter(other.items()[i+1][1])
        if 'core.lstm_coarse.weight_ih' in other.items()[i][0]:
            print("Initialize core.lstm0")
            del model.core.lstm0.weight_ih
            del model.core.lstm0.weight_hh
            del model.core.lstm0.bias_ih
            del model.core.lstm0.bias_hh
            model.core.lstm0.weight_ih= nn.Parameter(other.items()[i][1])
            model.core.lstm0.weight_hh = nn.Parameter(other.items()[i + 1][1])
            model.core.lstm0.bias_ih = nn.Parameter(other.items()[i + 2][1])
            model.core.lstm0.bias_hh = nn.Parameter(other.items()[i + 3][1])
        if 'core.lstm_fine_0.weight_ih' in other.items()[i][0]:
            print("Initialize core.lstm1")
            del model.core.lstm1.weight_ih
            del model.core.lstm1.weight_hh
            del model.core.lstm1.bias_ih
            del model.core.lstm1.bias_hh
            model.core.lstm1.weight_ih = nn.Parameter(other.items()[i][1])
            model.core.lstm1.weight_hh = nn.Parameter(other.items()[i + 1][1])
            model.core.lstm1.bias_ih = nn.Parameter(other.items()[i + 2][1])
            model.core.lstm1.bias_hh = nn.Parameter(other.items()[i + 3][1])
        if 'core.lstm_fine_1.weight_ih' in other.items()[i][0]:
            print("Initialize core.lstm2")
            del model.core.lstm2.weight_ih
            del model.core.lstm2.weight_hh
            del model.core.lstm2.bias_ih
            del model.core.lstm2.bias_hh
            model.core.lstm2.weight_ih = nn.Parameter(other.items()[i][1])
            model.core.lstm2.weight_hh = nn.Parameter(other.items()[i + 1][1])
            model.core.lstm2.bias_ih = nn.Parameter(other.items()[i + 2][1])
            model.core.lstm2.bias_hh = nn.Parameter(other.items()[i + 3][1])
        if 'core.attention.h2att.weight' in other.items()[i][0]:
            print("Initialize core.att1")
            del model.core.att1.h2att.weight
            del model.core.att1.h2att.bias
            del model.core.att1.alpha_net.weight
            del model.core.att1.alpha_net.bias
            model.core.att1.h2att.weight = nn.Parameter(other.items()[i][1])
            model.core.att1.h2att.bias = nn.Parameter(other.items()[i + 1][1])
            model.core.att1.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
            model.core.att1.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
            print("Initialize core.att2")
            del model.core.att2.h2att.weight
            del model.core.att2.h2att.bias
            del model.core.att2.alpha_net.weight
            del model.core.att2.alpha_net.bias
            model.core.att2.h2att.weight = nn.Parameter(other.items()[i][1])
            model.core.att2.h2att.bias = nn.Parameter(other.items()[i + 1][1])
            model.core.att2.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
            model.core.att2.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
    '''
    other = torch.load('save/20180102-001603.stackcap/model-best.pth')
    for i in range(len(other.items())):
        if 'embed.0.weight' in other.items()[i][0]:
            print("Initialize embed")
            del model.embed[0].weight
            model.embed[0].weight = nn.Parameter(other.items()[i][1])
        if 'fc_embed.0.weight' in other.items()[i][0]:
            print("Initialize fc_embed")
            del model.fc_embed[0].weight
            del model.fc_embed[0].bias
            model.fc_embed[0].weight = nn.Parameter(other.items()[i][1])
            model.fc_embed[0].bias = nn.Parameter(other.items()[i + 1][1])
        if 'att_embed.0.weight' in other.items()[i][0] and False:
            print("Initialize att_embed")
            del model.att_embed[1].weight
            del model.att_embed[1].bias
            model.att_embed[1].weight = nn.Parameter(other.items()[i][1])
            model.att_embed[1].bias = nn.Parameter(other.items()[i + 1][1])
        if 'logit.weight' in other.items()[i][0]:
            print("Initialize logit")
            del model.logit.weight
            del model.logit.bias
            model.logit.weight = nn.Parameter(other.items()[i][1])
            model.logit.bias = nn.Parameter(other.items()[i + 1][1])
        if 'ctx2att.weight' in other.items()[i][0] and False:
            print("Initialize ctx2att")
            del model.ctx2att.weight
            del model.ctx2att.bias
            model.ctx2att.weight = nn.Parameter(other.items()[i][1])
            model.ctx2att.bias = nn.Parameter(other.items()[i + 1][1])
        if 'core.att1.h2att.weight' in other.items()[i][0] and False:
            print("Initialize core.att1")
            del model.core.att1.h2att.weight
            del model.core.att1.h2att.bias
            del model.core.att1.alpha_net.weight
            del model.core.att1.alpha_net.bias
            model.core.att1.h2att.weight = nn.Parameter(other.items()[i][1])
            model.core.att1.h2att.bias = nn.Parameter(other.items()[i + 1][1])
            model.core.att1.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
            model.core.att1.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
            print("Initialize core.att2")
            del model.core.att2.h2att.weight
            del model.core.att2.h2att.bias
            del model.core.att2.alpha_net.weight
            del model.core.att2.alpha_net.bias
            model.core.att2.h2att.weight = nn.Parameter(other.items()[i][1])
            model.core.att2.h2att.bias = nn.Parameter(other.items()[i + 1][1])
            model.core.att2.alpha_net.weight = nn.Parameter(other.items()[i + 2][1])
            model.core.att2.alpha_net.bias = nn.Parameter(other.items()[i + 3][1])
        if 'core.lstm0.weight_ih' in other.items()[i][0]:
            print("Initialize core.lstm0")
            del model.core.lstm0.weight_ih
            del model.core.lstm0.weight_hh
            del model.core.lstm0.bias_ih
            del model.core.lstm0.bias_hh
            model.core.lstm0.weight_ih = nn.Parameter(other.items()[i][1])
            model.core.lstm0.weight_hh = nn.Parameter(other.items()[i + 1][1])
            model.core.lstm0.bias_ih = nn.Parameter(other.items()[i + 2][1])
            model.core.lstm0.bias_hh = nn.Parameter(other.items()[i + 3][1])
        if 'core.lstm1.weight_ih' in other.items()[i][0]:
            print("Initialize core.lstm1")
            del model.core.lstm1.weight_ih
            del model.core.lstm1.weight_hh
            del model.core.lstm1.bias_ih
            del model.core.lstm1.bias_hh
            model.core.lstm1.weight_ih = nn.Parameter(other.items()[i][1])
            model.core.lstm1.weight_hh = nn.Parameter(other.items()[i + 1][1])
            model.core.lstm1.bias_ih = nn.Parameter(other.items()[i + 2][1])
            model.core.lstm1.bias_hh = nn.Parameter(other.items()[i + 3][1])
        if 'core.lstm2.weight_ih' in other.items()[i][0]:
            print("Initialize core.lstm2")
            del model.core.lstm2.weight_ih
            del model.core.lstm2.weight_hh
            del model.core.lstm2.bias_ih
            del model.core.lstm2.bias_hh
            model.core.lstm2.weight_ih = nn.Parameter(other.items()[i][1])
            model.core.lstm2.weight_hh = nn.Parameter(other.items()[i + 1][1])
            model.core.lstm2.bias_ih = nn.Parameter(other.items()[i + 2][1])
            model.core.lstm2.bias_hh = nn.Parameter(other.items()[i + 3][1])
        if 'core.emb2.weight' in other.items()[i][0]:
            print("Initialize core.emb2")
            del model.core.emb2.weight
            del model.core.emb2.bias
            model.core.emb2.weight = nn.Parameter(other.items()[i][1])
            model.core.emb2.bias = nn.Parameter(other.items()[i + 1][1])
        if 'core.fusion1.0.weight' in other.items()[i][0]:
            print("Initialize core.fusion1")
            del model.core.fusion1[0].weight
            del model.core.fusion1[0].bias
            del model.core.fusion2[0].weight
            del model.core.fusion2[0].bias
            model.core.fusion1[0].weight = nn.Parameter(other.items()[i][1])
            model.core.fusion1[0].bias = nn.Parameter(other.items()[i + 1][1])
            model.core.fusion2[0].weight = nn.Parameter(other.items()[i + 2][1])
            model.core.fusion2[0].bias = nn.Parameter(other.items()[i + 3][1])

    return model
