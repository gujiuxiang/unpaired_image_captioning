from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import subprocess
import nltk
import torch
import torch.nn as nn
from misc.dataloader import *
from six.moves import cPickle
import eval_utils
import misc.criterion as criterion
import misc.utils as utils
import models
import models.NMT_Models as NMT_Models
from misc.dataloader.dataloaderraw import *
from misc.dataloader.dataloader import *
from misc.dataloader.dataloader_coco import *
from misc.dataloader.dataloader_aic import *
from misc.dataloader.dataloaderraw_coco import *
from eval_utils import *

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--dataset', type=str, default='aic_i2c', help='aic_i2c, mscoco')
parser.add_argument('--coco_eval_flag', type=int, default=0, help='eval nmt enable')
parser.add_argument('--model', type=str, default='save/20180602-152023.transformer/model_i2t-best.pth', help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101', help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='', help='path to infos to evaluate')
parser.add_argument('--input_nmt_data', default='data/nmt/demo_1000k.train.pt', help='Path to the *-train.pt file from preprocess.py')
# Basic options
parser.add_argument('--batch_size', type=int, default=10, help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=5000, help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0, help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1, help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0, help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1, help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0, help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=2, help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1, help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5, help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0, help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_json', type=str, default='data/aic_i2t/chinese_talk.json', help='path to the json file containing additional info and vocab')
parser.add_argument('--input_fc_dir', type=str, default='data/aic_i2t/bu_data/bu_fc', help='path to the directory containing the preprocessed fc feats')
parser.add_argument('--input_att_dir', type=str, default='data/aic_i2t/bu_data/bu_att', help='path to the directory containing the preprocessed att feats')
parser.add_argument('--input_box_dir', type=str, default='data/aic_i2t/bu_data/bu_box', help='path to the directory containing the boxes of att feats')
parser.add_argument('--input_box_cls_prob_dir', type=str, default='data/aic_i2t/bu_data/bu_box_cls_prob', help='path to the directory containing the boxes of att feats')
parser.add_argument('--input_box_keep_boxes_dir', type=str, default='data/aic_i2t/bu_data/bu_box_keep_boxes', help='path to the directory containing the boxes of att feats')
parser.add_argument('--input_label_h5', type=str, default='data/aic_i2t/chinese_talk_label.h5', help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--split', type=str, default='test', help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')

# misc
parser.add_argument('--id', type=str, default='', help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1, help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, help='if we need to calculate loss.')

opt = parser.parse_args()

# Load infos
opt.infos_path = os.path.join(opt.model.split('/')[0], opt.model.split('/')[1], 'infos-best.pkl')
opt.id = opt.model.split('/')[1]
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = 'data/' + opt.dataset + '/bu_data/bu_fc'
    opt.input_att_dir = 'data/' + opt.dataset + '/bu_data/bu_att'
    opt.input_box_dir = 'data/' + opt.dataset + '/bu_data/bu_box'
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval",
          "input_json", "input_fc_dir", "input_att_dir", "input_box_dir",
          "input_box_cls_prob_dir", "input_box_keep_boxes_dir",
          "input_label_h5", "coco_eval_flag"]

for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

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

def eval_coco_online(mscoco_zh_text):
    from googletrans import Translator
    from time import sleep
    # Read in the file
    file = open(mscoco_zh_text, 'r')
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

    tmp_name = mscoco_zh_text.replace('zh', 'en').replace('.txt', 'offline')
    with open(tmp_name + '.txt', 'w') as file:
        for line in en_lines:
            file.write("%s\n" % line.encode("utf-8").lower())
    print('Calculating scores for the generated results ... ...')
    utils.text2textid(tmp_name, mscoco_zh_text.replace('.txt', '.json'))
    utils.test2cocojson(tmp_name + '_id')
    lang_stats = language_eval('coco', json.load(open(tmp_name + '_id.json')), opt.id, '')

def eval_coco_offline():
    dump_json_path = i2t_eval()
    root = os.getcwd() + '/'
    use_translation = False
    mscoco_zh_json = dump_json_path
    mscoco_zh_text = dump_json_path.replace('.json', '.txt')
    utils.cocojson2text(mscoco_zh_json, mscoco_zh_json.replace('.json', '.txt'))
    nmt_model = root + "save/opennmt/aic_zh2en_nmt_part_acc_61.49_ppl_7.43_e22.pt"
    print("Start translating chinese to english ...")
    bashCommand = "cd ../OpenNMT-py && python translate.py" + \
                  " -model " + nmt_model + \
                  " -src " + root + '/' + mscoco_zh_text + \
                  " -output " + root + '/' + mscoco_zh_text.replace('zh', 'en') + \
                  " -verbose -gpu 0"
    _output = subprocess.check_output(['bash', '-c', bashCommand])
    print("Finish translating chinese to english ...")

    self_bleu_flag = False
    en_lines = []
    if self_bleu_flag:
        print('------------------------------------------- aic self-bleu')
        count = 0
        file = open(mscoco_zh_text.replace('zh', 'en'), 'r')
        zh_lines = file.readlines()
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
        tmp_name = mscoco_zh_text.replace('zh', 'en').replace('.txt', 'offline')
        file = open(mscoco_zh_text.replace('zh', 'en'), 'r')
        en_lines = file.readlines()
        with open(tmp_name + '.txt', 'w') as file:
            for line in en_lines:
                file.write("%s" % line.encode("utf-8").lower())
        utils.text2textid(tmp_name, dump_json_path)
        print('Calculating scores for the generated results ... ...')
        utils.test2cocojson(tmp_name + '_id')
        lang_stats = language_eval('coco', json.load(open(tmp_name + '_id.json')), opt.id, '')

def i2t_eval():
    # Setup the model
    i2t_model = models.setup(opt)
    i2t_model.load_state_dict(torch.load(opt.model))
    i2t_model.cuda()
    i2t_model.eval()
    crit = criterion.LanguageModelCriterion(infos['opt'])

    # Create the Data Loader instance
    if len(opt.image_folder) == 0:
        loader = DataLoader(opt)
    else:
        loader = DataLoaderRaw({'folder_path': opt.image_folder,
                                'coco_json': opt.coco_json,
                                'batch_size': opt.batch_size,
                                'cnn_model': opt.cnn_model})
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.ix_to_word = infos['vocab']

    # Set sample options
    loss, split_predictions, lang_stats, nmt_valid_ppl, nmt_valid_acc =eval_utils.eval_split(opt, loader, i2t_model, None, vars(opt))

    print('loss: ', loss)
    if lang_stats:
        print(lang_stats)

    if opt.dump_json == 1:
        # dump the json
        dump_json_path = 'tmp/' + opt.model.split('/')[1] + '_zh_' + opt.dataset + '.json'
        json.dump(split_predictions, open('tmp/' + opt.model.split('/')[1] + '_zh_' + opt.dataset + '.json', 'w'))

    return dump_json_path

def i2t_nmt_eval(opt_eval, opt_coco):
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
    nmt_model.generator = nn.Sequential(nn.Linear(opt_eval.rnn_size, tgt_dict.size()), nn.LogSoftmax())
    nmt_model.load_state_dict(torch.load(opt_eval.nmt_model, map_location=lambda storage, loc: storage))
    nmt_model.cuda()
    nmt_model.eval()
    print('* number of parameters: %d' % sum([p.nelement() for p in nmt_model.parameters()]))
    print('Current model parameters\n')

    loader = DataLoader(opt_eval)
    mscoco_loader = DataLoader_COCO(opt_coco)
    loader.ix_to_word = infos['vocab']
    i2t_run = True
    i2t_val_loss, predictions, coco_predictions, _lang_stats, _coco_lang_stats, nmt_valid_ppl, nmt_valid_acc = eval_utils.eval_split_coco(opt_eval, loader, mscoco_loader, i2t_model, nmt_model, vars(opt_eval))

if __name__ == "__main__":
    eval_coco_offline()