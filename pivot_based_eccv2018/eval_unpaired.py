from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import nltk
import math
import json
import argparse
import hashlib
import torch
import torch.nn as nn
from misc.dataloader import *
import models.NMT_Models as NMT_Models
from six.moves import cPickle
import opts
import eval_utils
from misc.resnet_utils import myResnet
import misc.resnet as resnet

import misc.criterion as criterion
import misc.utils as utils
from models.weight_init import *
import models
from misc.dataloader.dataloaderraw import *
from misc.dataloader.dataloader import *
from misc.dataloader.dataloader_coco import *
from misc.dataloader.dataloader_aic import *
from misc.dataloader.dataloaderraw_coco import *
from misc.nmt_translator import *
import os
import subprocess
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from abc import abstractmethod

# Input arguments and options
parser_coco = argparse.ArgumentParser()
parser_coco.add_argument('--model_name', type=str, default='fc', help='')
parser_coco.add_argument('--start_from', type=str, default='save/20180305-115902.fcnmt_True', help='None')
parser_coco.add_argument('--ensemble', type=int, default=1)  # 'use pool to load pre-extracted features from dir'
parser_coco.add_argument('--data_type', type=int, default=1)  # 'use pool to load pre-extracted features from dir'
# Model settings
parser_coco.add_argument('--exchange_vocab', type=int, default=0)
# Input paths
parser_coco.add_argument('--disp_coarse_to_fine', type=int, default=0)
parser_coco.add_argument('--dump_output', type=int, default=0)
parser_coco.add_argument('--shuffle', type=int, default=0)#load shuffled data from h5 or dir
parser_coco.add_argument('--pool_loader', type=int, default=0)#use pool to load pre-extracted features from dir
parser_coco.add_argument('--pre_ft', type=int, default=1)#pre extracted feature input
# Basic options
parser_coco.add_argument('--batch_size', type=int, default=100)#if > 0 then overrule, otherwise load from checkpoint.
parser_coco.add_argument('--val_images_use', type=int, default=5000)#how many images to use when periodically evaluating the loss? (-1 = all)
parser_coco.add_argument('--language_eval', type=int, default=0)
parser_coco.add_argument('--dump_images', type=int, default=0) #Dump images into vis/imgs folder for vis? (1=yes,0=no)
parser_coco.add_argument('--dump_json', type=int, default=1) #Dump json with predictions into vis folder? (1=yes,0=no)
parser_coco.add_argument('--dump_path', type=int, default=0) #Write image paths along with predictions into vis json? (1=yes,0=no)
# Sampling options
parser_coco.add_argument('--sample_max', type=int, default=1) #1 = sample argmax words. 0 = sample from distributions.
parser_coco.add_argument('--beam_size', type=int, default=5)
parser_coco.add_argument('--temperature', type=float, default=1.0)
# For evaluation on a folder of images:
parser_coco.add_argument('--cnn_model', type=str, default='resnet101', help='resnet101, resnet152')
parser_coco.add_argument('--feature_type', type=str, default='resnet101')
parser_coco.add_argument('--fc_feat_size', type=int, default=2048) # '2048 for resnet, 4096 for vgg'
parser_coco.add_argument('--att_feat_size', type=int, default=2048) # '2048 for resnet, 512 for vgg'
parser_coco.add_argument('--att_hid_size', type=int, default=512)#the hidden size of the attention MLP;0 if not using hidden layer')
parser_coco.add_argument('--seq_per_img', type=int, default=5)
parser_coco.add_argument('--server', type=int, default=0)#if running on MSCOCO challenge
parser_coco.add_argument('--server_best', type=int, default=1)#if running on MSCOCO challenge
parser_coco.add_argument('--split', type=str, default='val')#if running on MSCOCO images, which split to use: val|test|train
parser_coco.add_argument('--coco_json', type=str, default='/home/jxgu/github/im2text_jxgu/pytorch/data/mscoco/image_info_karpathy_5k_test.json')#if nonempty then use this file in DataLoaderRaw (see docs there)
parser_coco.add_argument('--image_folder', type=str, default='/home/jxgu/github/im2text_jxgu/pytorch/data/mscoco')#If this is nonempty then will predict on the images in this folder path
parser_coco.add_argument('--image_root', type=str, default='/home/jxgu/github/im2text_jxgu/pytorch/data/mscoco') #In case the image paths have to be preprended with a root path to an image folder
# For evaluation on a folder of h5 file:
parser_coco.add_argument('--input_nmt_dict', default='data/ai_challenger/machine_translation/nmt_t2t_data_all/nmt_all_0210.dicts.pt', help='')
parser_coco.add_argument('--input_coco_json', type=str, default='data/mscoco/cocotalk_karpathy.json')
parser_coco.add_argument('--input_fc_dir', type=str, default='data/mscoco/cocobu_fc', help='path to the h5file containing the preprocessed dataset')
parser_coco.add_argument('--input_att_dir', type=str, default='data/mscoco/cocobu_att', help='path to the h5file containing the preprocessed dataset')
parser_coco.add_argument('--input_box_dir', type=str, default='data/mscoco/cocobu_box', help='path to the h5file containing the preprocessed dataset')

parser_coco.add_argument('--input_im_h5', type=str, default='data/ai_challenger/cocotalk_karpathy_images.h5')
parser_coco.add_argument('--input_fc_coco_h5', type=str, default='data/mscoco/cocotalk_karpathy_fc_0816.h5')
parser_coco.add_argument('--input_att_coco_h5', type=str, default='data/mscoco/cocotalk_karpathy_att_0816.h5')
parser_coco.add_argument('--input_label_coco_h5', type=str, default='data/mscoco/cocotalk_karpathy_label_0816.h5')

# misc
parser_coco.add_argument('--model_id', type=str, default='')#an id identifying this run/job. used in cross-val and appended when writing progress files
parser_coco.add_argument('--id', type=str, default='evalscript')#an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files


# Input arguments and options
parser_nmt = argparse.ArgumentParser()
# Input paths
parser_nmt.add_argument('--i2t_model', type=str, default='', help='path to model to evaluate')
parser_nmt.add_argument('--nmt_model', type=str, default='', help='path to model to evaluate')
parser_nmt.add_argument('--cnn_model', type=str, default='resnet101', help='resnet101, resnet152')
parser_nmt.add_argument('--input_nmt_data', default='data/nmt/demo_1000k.train.pt', help='Path to the *-train.pt file from preprocess.py')
# Basic options
parser_nmt.add_argument('--batch_size', type=int, default=0, help='if > 0 then overrule, otherwise load from checkpoint.')
parser_nmt.add_argument('--num_images', type=int, default=-1, help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser_nmt.add_argument('--language_eval', type=int, default=0, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser_nmt.add_argument('--dump_images', type=int, default=0, help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser_nmt.add_argument('--dump_json', type=int, default=1, help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser_nmt.add_argument('--dump_path', type=int, default=0, help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser_nmt.add_argument('--sample_max', type=int, default=1, help='1 = sample argmax words. 0 = sample from distributions.')
parser_nmt.add_argument('--max_ppl', type=int, default=0, help='beam search by max perplexity or max probability.')
parser_nmt.add_argument('--beam_size', type=int, default=1, help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser_nmt.add_argument('--group_size', type=int, default=1, help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser_nmt.add_argument('--diversity_lambda', type=float, default=0.5, help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser_nmt.add_argument('--temperature', type=float, default=1.0, help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser_nmt.add_argument('--decoding_constraint', type=int, default=0, help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser_nmt.add_argument('--image_folder', type=str, default='', help='If this is nonempty then will predict on the images in this folder path')
parser_nmt.add_argument('--image_root', type=str, default='', help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser_nmt.add_argument('--input_json', type=str, default='data/ai_challenger/image_captioning/chinese_talk.json', help='path to the json file containing additional info and vocab')
parser_nmt.add_argument('--input_fc_dir', type=str, default='data/ai_challenger/image_captioning/chinese_bu_fc', help='path to the directory containing the preprocessed fc feats')
parser_nmt.add_argument('--input_att_dir', type=str, default='data/ai_challenger/image_captioning/chinese_bu_att', help='path to the directory containing the preprocessed att feats')
parser_nmt.add_argument('--input_box_dir', type=str, default='data/ai_challenger/image_captioning/chinese_bu_box', help='path to the directory containing the boxes of att feats')
parser_nmt.add_argument('--input_label_h5', type=str, default='data/ai_challenger/image_captioning/chinese_talk_label.h5', help='path to the h5file containing the preprocessed dataset')
parser_nmt.add_argument('--input_fc_h5', type=str, default='data/ai_challenger/image_captioning/chinese_talk_1030_resnet101_fc.h5')
parser_nmt.add_argument('--input_att_h5', type=str, default='data/ai_challenger/image_captioning/chinese_talk_1030_resnet101_att.h5')
parser_nmt.add_argument('--split', type=str, default='test', help='if running on MSCOCO images, which split to use: val|test|train')
parser_nmt.add_argument('--coco_json', type=str, default='', help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')

NUM_THREADS = 2  # int(os.environ['OMP_NUM_THREADS'])

def language_eval_json(filename):
    preds = json.load(open(filename+'_id.json', 'r'))
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    #encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join(filename+'_id_tmp.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def language_eval_json_30K(filename):
    preds = json.load(open(filename+'_id.json', 'r'))
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/flickr30k_val.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    #encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join(filename+'_id_tmp.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    valids = map(int, valids)
    preds_filt = [{'caption': p['caption'], 'image_id': str(p['image_id'])} for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def language_eval(type, preds, model_id, split):
    import sys
    if type == 'en':
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    elif type == 'en_30K':
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/flickr30k_val.json'
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    else:
        sys.path.append("AI_Challenger/Evaluation/caption_eval")
        annFile = 'data/ai_challenger/image_captioning/eval_reference.json'
        from coco_caption.pycxtools.coco import COCO
        from coco_caption.pycxevalcap.eval import COCOEvalCap

    #encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    if type == 'en':
        preds_filt = [p for p in preds if p['image_id'] in valids]
        print('using %d/%d predictions' % (len(preds_filt), len(preds)))
        json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...
    else:
        json.dump(preds, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    if type == 'en':
        for p in preds_filt:
            image_id, caption = p['image_id'], p['caption']
            imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def coco_fc_ext():
    test = json.load(open('tmp/08170034_cnn_resnet101.lm_debug5_scst_.rnn_LSTM_test.json', 'r'))
    caps = []
    for k in test['imgToEval'].keys(): caps.append(test['imgToEval'][k]['caption'])

    with open('tmp/coco_i2t_fc_outs.txt', 'w') as file:
        for line in caps:
            file.write("%s\n" % line.encode("utf-8").lower())
    reference = utils.get_reference_all('tmp/coco_i2t_fc_outs.txt')
    return reference

def self_BLEU(in_text_file):
    print('------------------------------------------- self-bleu')
    reference = utils.get_reference_single(in_text_file)
    for i in range(7):
        print('Self BLEU: {}'.format(utils.self_bleu(reference, i)))
    print('------------------------------------------- end')

def eval_30K(type, text_in):
    import time
    import re, string
    if type == 'online':
        from googletrans import Translator
        translator = Translator()

    # Read in the file
    file = open(text_in, 'r')
    zh_lines = file.readlines()
    en_lines = []
    count = 0
    for line in zh_lines:
        if type == 'online':
            translation = translator.translate(line)
            #en_lines.append(translation.text.replace("there is", ""))
            tmp = translation.text
            #tmp = re.sub('[%s]' % re.escape(string.punctuation), '', tmp)
        else:
            tmp = line.replace("there is", "")
        en_lines.append(tmp)
        count += 1
        if count % 100 == 0:
            if type == 'online':
                translator = Translator()
            print('... %d sentences prepared' % count)

    tmp_name = 'tmp/flickr_test_1k_en_' + type
    with open(tmp_name + '.txt', 'w') as file:
        for line in en_lines:
            file.write("%s\n" % line.encode("utf-8").lower())

    ref_id_json = 'tmp/captions_val_image_info_karpathy_1k_test_11080899_results.json'
    utils.text2textid(tmp_name, ref_id_json)
    print('Calculating scores for the generated results ... ...')
    utils.test2cocojson(tmp_name + '_id')
    lang_stats = language_eval_json_30K(tmp_name)

def eval_coco_online():

    from googletrans import Translator
    from time import sleep

    # Read in the file
    file = open('/home/jxgu/github/im2text_jxgu/pytorch/tmp/coco_test_5k_zh.txt', 'r')
    zh_lines = file.readlines()
    en_lines = []
    count = 0
    translator = Translator()
    for line in zh_lines:
        translation = translator.translate(line)
        #en_lines.append(translation.text.replace("there is", ""))
        en_lines.append(translation.text)
        count += 1
        if count % 100 == 0:
            translator = Translator()
            print('... %d sentences prepared' % count)

    tmp_name = 'tmp/coco_test_5k_en_online'
    with open(tmp_name + '.txt', 'w') as file:
        for line in en_lines:
            file.write("%s\n" % line.encode("utf-8").lower())


    ref_id_json = 'tmp/captions_val_image_info_karpathy_5k_test_11080899_results.json'
    utils.text2textid(tmp_name, ref_id_json)
    print('Calculating scores for the generated results ... ...')
    utils.test2cocojson(tmp_name + '_id')
    lang_stats = language_eval_json(tmp_name)

def eval_coco_offline():
    #os.system("bash test.sh")
    from googletrans import Translator
    translator = Translator()
    root = os.getcwd() + '/'
    use_translation = False
    mscoco_src_text = '/home/jxgu/github/im2text_jxgu/pytorch/tmp/coco_test_5k_zh.txt'
    if os.path.exists(mscoco_src_text) is True and use_translation:
        #mscoco_zh_json = 'tmp/20180419-075726.denseatt_zh_mscoco.json'
        #utils.cocojson2text(mscoco_zh_json, mscoco_zh_json.replace('.json', '.txt'))
        nmt_model = root + "neural_machine_translation/save/20180308-091231/demo-model-0303-full_acc_54.75_ppl_9.10_e22.pt"
        print("Start translating chinese to english ...")
        bashCommand = "cd neural_machine_translation && python translate.py" + \
                      " -model " + nmt_model + \
                      " -src " + mscoco_src_text + \
                      " -output " + root + 'tmp/coco_test_5k_en.txt' \
                      " -verbose -gpu 0"
        _output = subprocess.check_output(['bash', '-c', bashCommand])
        print("Finish translating chinese to english ...")

    self_bleu_flag = False
    en_lines = []
    print('------------------------------------------- aic self-bleu')
    if self_bleu_flag:
        file = open('tmp/coco_test_5k_zh_en.txt', 'r')
        zh_lines = file.readlines()
        count = 0
        for line in zh_lines:
            translation = line
            en_lines.append(translation.replace("there is", ""))
            count += 1
            if count % 100 == 0:
                print('... %d sentences prepared' % count)
        reference = list()
        for line in en_lines:
            text = nltk.word_tokenize(line)
            reference.append(text)
        for i in range(7):
            print('Self BLEU: {}'.format(utils.self_bleu(reference, i)))
    else:
        tmp_name = 'tmp/coco_test_5k_en_offline'
        file = open('tmp/coco_test_5k_zh.en.txt', 'r')
        en_lines = file.readlines()
        with open(tmp_name + '.txt', 'w') as file:
            for line in en_lines:
                file.write("%s" % line.encode("utf-8").lower())
        ref_id_json = 'tmp/captions_val_image_info_karpathy_5k_test_11080899_results.json'
        utils.text2textid(tmp_name, ref_id_json)
        print('Calculating scores for the generated results ... ...')
        utils.test2cocojson(tmp_name + '_id')
        lang_stats = language_eval_json(tmp_name)
    print('------------------------------------------- end')

def eval(opt_eval, opt_coco):

    # Load infos
    opt_eval.batch_size = opt_coco.batch_size
    opt_eval.i2t_model = opt_coco.start_from + '/model_i2t-best.pth'
    opt_eval.nmt_model = opt_coco.start_from + '/model_nmt-best.pth'
    opt_eval.infos_path = os.path.join(opt_eval.i2t_model.split('/')[0], opt_eval.i2t_model.split('/')[1], 'infos-best.pkl')
    opt_eval.id = opt_eval.i2t_model.split('/')[1]
    with open(opt_eval.infos_path) as f:
        infos = cPickle.load(f)

    # override and collect parameters
    if len(opt_eval.input_fc_dir) == 0:
        opt_eval.input_fc_dir = infos['opt'].input_fc_dir
        opt_eval.input_att_dir = infos['opt'].input_att_dir
        opt_eval.input_att_dir = infos['opt'].input_box_dir
        opt_eval.input_label_h5 = infos['opt'].input_label_h5
    if len(opt_eval.input_json) == 0:
        opt_eval.input_json = infos['opt'].input_json
    if opt_eval.batch_size == 0:
        opt_eval.batch_size = infos['opt'].batch_size
    if len(opt_eval.id) == 0:
        opt_eval.id = infos['opt'].id
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval",
              "input_json", "input_fc_dir", "input_att_dir", "input_box_dir",
              "input_fc_h5", "input_att_h5", "input_label_h5"]

    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt_eval):
                assert vars(opt_eval)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt_eval).update({k: vars(infos['opt'])[k]})  # copy over options from model

    vocab = infos['vocab']  # ix -> word mapping

    # Setup the model
    i2t_model = models.setup(opt_eval)
    i2t_model.load_state_dict(torch.load(opt_eval.i2t_model))
    i2t_model.cuda()
    i2t_model.eval()

    # teacher encoder and decoder
    nmt_dicts = torch.load(opt_eval.input_nmt_dict)['dicts']
    src_dict = nmt_dicts['src']
    tgt_dict = nmt_dicts['tgt']
    nmt_encoder = NMT_Models.Encoder(opt_eval, src_dict)
    nmt_decoder = NMT_Models.Decoder(opt_eval, tgt_dict)
    nmt_model = NMT_Models.NMTModel(opt_eval, nmt_encoder, nmt_decoder, src_dict, tgt_dict, len(opt_eval.gpus) > 1)
    nmt_generator = nn.Sequential(nn.Linear(opt_eval.rnn_size, tgt_dict.size()), nn.LogSoftmax())
    
    checkpoint = torch.load(opt_eval.nmt_model, map_location=lambda storage, loc: storage)
    #nmt_model.load_state_dict(checkpoint['model'])
    #nmt_generator.load_state_dict(checkpoint['generator'])
    nmt_model.generator = nmt_generator
    nmt_model.load_state_dict(checkpoint)
    print('* number of parameters: %d' % sum([p.nelement() for p in nmt_model.parameters()]))
    print('Current model parameters\n')
    #print(nmt_model)
    nmt_model.cuda()
    nmt_model.eval()

    loader = DataLoader(opt_eval)
    mscoco_loader = DataLoader_COCO(opt_coco)
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']
    # Set sample options
    i2t_run = True

    i2t_val_loss, predictions, coco_predictions, _lang_stats, _coco_lang_stats, nmt_valid_ppl, nmt_valid_acc = eval_utils.eval_split_coco(opt_eval, loader, mscoco_loader, i2t_model, nmt_model, vars(opt_eval))


if __name__ == "__main__":
    opt_coco = parser_coco.parse_args()
    opt_eval = parser_nmt.parse_args()
    #eval(opt_eval, opt_coco)
    #eval_coco_online()
    eval_coco_offline()
    #offline_eval_30K()
    #eval_30K('online', '/home/jxgu/github/im2text_jxgu/pytorch/tmp/flickr_test_1k_zh.txt')
    #create_zh_en_html()
    #utils.visulize_dep([u"the bright room has a man in a white shirt standing here", u"the blue sky has a man with a hat on the railing"])
    #utils.quicktree("This is a parse tree.")
