# -*- coding: utf-8 -*-
#from pycontractions import Contractions
import random
import contractions
import onmt
import onmt.Markdown
import onmt.IO
import argparse
import torch
import codecs
import h5py
import pdb
import string
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zhon.hanzi import punctuation

parser = argparse.ArgumentParser(description='preprocess.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-src_img_dir', default=".",
                    help="Location of source images")


parser.add_argument('-train_tgt', default='/home/jxgu/github/unparied_im2text_jxgu/data/aic_mt/nmt_t2t_data_all/train_0303.zh',
                    help="Path to the training source data")
parser.add_argument('-train_src', default='/home/jxgu/github/unparied_im2text_jxgu/data/aic_mt/nmt_t2t_data_all/train_0303.en',
                    help="Path to the training target data")
parser.add_argument('-valid_tgt', default='/home/jxgu/github/unparied_im2text_jxgu/data/aic_mt/nmt_t2t_data_all/valid_0303.zh',
                    help="Path to the validation source data")
parser.add_argument('-valid_src', default='/home/jxgu/github/unparied_im2text_jxgu/data/aic_mt/nmt_t2t_data_all/valid_0303.en',
                    help="Path to the validation target data")

parser.add_argument('-save_data', default='/home/jxgu/github/unparied_im2text_jxgu/data/aic_mt/nmt_t2t_data_all/nmt_all_0303_bt',
                    help="Output file for the prepared data")
parser.add_argument('-write_txt', action='store_true',
                    help="Write training files in txt format")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")
parser.add_argument('-features_vocabs_prefix', type=str, default='',
                    help="Path prefix to existing features vocabularies")
parser.add_argument('-src_seq_length_min', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length_min', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_seq_length', type=int, default=50,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    "Construct the word and feature vocabs."
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)
    featuresVocabs = []
    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            words, features, numFeatures = extractFeatures(sent.split())

            if len(featuresVocabs) == 0 and numFeatures > 0:
                for j in range(numFeatures):
                    featuresVocabs.append(onmt.Dict([onmt.Constants.PAD_WORD,
                                                     onmt.Constants.UNK_WORD,
                                                     onmt.Constants.BOS_WORD,
                                                     onmt.Constants.EOS_WORD]))
            else:
                assert len(featuresVocabs) == numFeatures, \
                    "all sentences must have the same number of features"

            for i in range(len(words)):
                vocab.add(words[i])
                for j in range(numFeatures):
                    featuresVocabs[j].add(features[j][i])

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab, featuresVocabs


def initVocabulary(name, dataFile, vocabFile, vocabSize):
    """If `vocabFile` exists, read it in,
    Else, generate from data."""
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab, genFeaturesVocabs = makeVocabulary(dataFile, vocabSize)
        vocab = genWordVocab
        featuresVocabs = genFeaturesVocabs

    print()
    return vocab, featuresVocabs


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def saveFeaturesVocabularies(name, vocabs, prefix):
    for j in range(len(vocabs)):
        file = prefix + '.' + name + '_feature_' + str(j) + '.dict'
        print('Saving ' + name + ' feature ' + str(j) +
              ' vocabulary to \'' + file + '\'...')
        vocabs[j].writeFile(file)

def extractFeatures(tokens):
    "Given a list of token separate out words and features (if any)."
    words = []
    features = []
    numFeatures = None

    for t in range(len(tokens)):
        field = tokens[t].split(u"￨")
        word = field[0]
        if len(word) > 0:
            words.append(word)

            if numFeatures is None:
                numFeatures = len(field) - 1
            else:
                assert (len(field) - 1 == numFeatures), \
                    "all words must have the same number of features"

            if len(field) > 1:
                for i in range(1, len(field)):
                    if len(features) <= i-1:
                        features.append([])
                    features[i - 1].append(field[i])
                    assert (len(features[i - 1]) == len(words))
    return words, features, numFeatures if numFeatures else 0

def readSrcLine(src_line, src_dict, src_feature_dicts,
                _type="text", src_img_dir=""):
    srcFeats = None
    srcWords, srcFeatures, _ = extractFeatures(src_line)
    srcData = src_dict.convertToIdx(srcWords,
                                    onmt.Constants.UNK_WORD)
    if src_feature_dicts:
        srcFeats = [src_feature_dicts[j].
                        convertToIdx(srcFeatures[j],
                                     onmt.Constants.UNK_WORD)
                    for j in range(len(src_feature_dicts))]

    return srcWords, srcData, srcFeats

def readTgtLine(tgt_line, tgt_dict, tgt_feature_dicts, _type="text"):
    tgtFeats = None
    tgtWords, tgtFeatures, _ = extractFeatures(tgt_line)
    tgtData = tgt_dict.convertToIdx(tgtWords,
                                    onmt.Constants.UNK_WORD,
                                    onmt.Constants.BOS_WORD,
                                    onmt.Constants.EOS_WORD)
    if tgt_feature_dicts:
        tgtFeats = [tgt_feature_dicts[j].
                    convertToIdx(tgtFeatures[j],
                                 onmt.Constants.UNK_WORD)
                    for j in range(len(tgt_feature_dicts))]

    return tgtWords, tgtData, tgtFeats

def makeData(srcFile, tgtFile, srcDicts, tgtDicts,
             srcFeatureDicts, tgtFeatureDicts):
    src, tgt = [], []
    srcFeats = [[] for i in range(len(srcFeatureDicts))]
    tgtFeats = [[] for i in range(len(tgtFeatureDicts))]
    alignments = []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, "r", "utf-8")
    tgtF = codecs.open(tgtFile, "r", "utf-8")

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcLine = sline.split()
        tgtLine = tline.split()

        if len(srcLine) <= opt.src_seq_length and len(srcLine) >= opt.src_seq_length_min  and len(tgtLine) <= opt.tgt_seq_length and len(tgtLine) >= opt.tgt_seq_length_min:

            # Check truncation condition.
            if opt.src_seq_length_trunc != 0:
                srcLine = srcLine[:opt.src_seq_length_trunc]

            if opt.tgt_seq_length_trunc != 0:
                tgtLine = tgtLine[:opt.tgt_seq_length_trunc]

            srcWords, srcData, srcFeat = readSrcLine(srcLine, srcDicts, srcFeatureDicts,_type=opt.src_type, src_img_dir=opt.src_img_dir)
            src += [srcData]
            for i in range(len(srcFeats)):
                srcFeats[i] += [srcFeat[i]]

            tgtWords, tgtData, tgtFeat = readTgtLine(tgtLine, tgtDicts,tgtFeatureDicts)
            tgt += [tgtData]
            for i in range(len(tgtFeats)):
                tgtFeats[i] += [tgtFeat[i]]

            alignments += [onmt.IO.align(srcWords, tgtWords)]
            sizes += [len(srcData)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        alignments = [alignments[idx] for idx in perm]
        for j in range(len(srcFeatureDicts)):
            srcFeats[j] = [srcFeats[j][idx] for idx in perm]
        for j in range(len(tgtFeatureDicts)):
            tgtFeats[j] = [tgtFeats[j][idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    alignments = [alignments[idx] for idx in perm]
    for j in range(len(srcFeatureDicts)):
        srcFeats[j] = [srcFeats[j][idx] for idx in perm]
    for j in range(len(tgtFeatureDicts)):
        tgtFeats[j] = [tgtFeats[j][idx] for idx in perm]

    print(('Prepared %d sentences ' +
          '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt, srcFeats, tgtFeats, alignments

def to_array(split_data, type):
    lengths = [x.size(0) for x in split_data[type]]
    max_length = max(lengths)
    min_length = min(lengths)
    print('Max length: {}, min length: {}'.format(max_length, min_length))
    N = len(split_data[type])
    label_arrays = []
    label_alignment = []
    #label_length = np.zeros(N, dtype='uint32')
    label_length = lengths
    #label_length = np.asarray(lengths).reshape(-1,1)
    label_ix = np.zeros(N, dtype='uint32')
    counter = 1
    for i in range(len(split_data[type])):
        n = len(split_data[type][i])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((1, max_length), dtype='uint32')
        for j in range(n):
            Li[0, j] = split_data[type][i].numpy()[j]

        # note: word indices are 1-indexed, and captions are padded with zeros
        counter += 1
        label_arrays.append(Li)

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    print('encoded captions to array of size ', L.shape)
    return L, label_ix, label_length

def to_array_alignments(split_data, type):
    lengths = [x.size(0) for x in split_data[type]]
    max_length = max(lengths)
    min_length = min(lengths)
    print('Max length: {}, min length: {}'.format(max_length, min_length))
    N = len(split_data[type])
    label_alignment = []
    counter = 1
    for i in range(len(split_data[type])):
        n = len(split_data[type][i])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((1, max_length), dtype='uint32')
        for j in range(n):
            Li[0, j] = split_data[type][i].numpy()[j]

        # note: word indices are 1-indexed, and captions are padded with zeros
        counter += 1
        label_alignment.append(Li)

    L = np.concatenate(label_alignment, axis=0)  # put all the labels together
    print('encoded captions to array of size ', L.shape)
    return L

def main_pt():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'], dicts['src_features'] = initVocabulary('source', opt.train_src, opt.src_vocab, opt.src_vocab_size)
    dicts['tgt'], dicts['tgt_features'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size)

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'], \
        valid['src_features'], valid['tgt_features'], \
        valid['alignments'] \
        = makeData(opt.valid_src, opt.valid_tgt,
                   dicts['src'], dicts['tgt'],
                   dicts['src_features'], dicts['tgt_features'])

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'], \
        train['src_features'], train['tgt_features'], \
        train['alignments'] \
        = makeData(opt.train_src, opt.train_tgt,
                   dicts['src'], dicts['tgt'],
                   dicts['src_features'], dicts['tgt_features'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')
    if opt.features_vocabs_prefix:
        saveFeaturesVocabularies('source', dicts['src_features'],
                                 opt.save_data)
        saveFeaturesVocabularies('target', dicts['tgt_features'],
                                 opt.save_data)


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.src_type,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')

def main_h5():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'], dicts['src_features'] = initVocabulary('source', opt.train_src, opt.src_vocab, opt.src_vocab_size)
    dicts['tgt'], dicts['tgt_features'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab, opt.tgt_vocab_size)

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'], \
        valid['src_features'], valid['tgt_features'], \
        valid['alignments'] \
        = makeData(opt.valid_src, opt.valid_tgt,
                   dicts['src'], dicts['tgt'],
                   dicts['src_features'], dicts['tgt_features'])

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'], \
        train['src_features'], train['tgt_features'], \
        train['alignments'] \
        = makeData(opt.train_src, opt.train_tgt,
                   dicts['src'], dicts['tgt'],
                   dicts['src_features'], dicts['tgt_features'])

    train_src_L, train_src_label_ix, train_src_label_length = to_array(train, 'src')
    train_tgt_L, train_tgt_label_ix, train_tgt_label_length = to_array(train, 'tgt')
    #train_alignments = to_array_alignments(train, 'alignments')
    print(train_src_L.shape)
    print(train_tgt_L.shape)
    valid_src_L, valid_src_label_ix, valid_src_label_length = to_array(valid, 'src')
    valid_tgt_L, valid_tgt_label_ix, valid_tgt_label_length = to_array(valid, 'tgt')
    #valid_alignments = to_array_alignments(valid, 'alignments')

    print('Saving data to \'' + opt.save_data + '.train.h5\'...')
    f_nmt = h5py.File(opt.save_data + '.train.h5', "w")

    f_nmt.create_dataset("train_src_label", dtype='uint32', data=train_src_L)
    f_nmt.create_dataset("train_src_label_start_ix", dtype='uint32', data=train_src_label_ix)
    f_nmt.create_dataset("train_src_label_length", dtype='uint32', data=train_src_label_length)
    f_nmt.create_dataset("train_tgt_label", dtype='uint32', data=train_tgt_L)
    f_nmt.create_dataset("train_tgt_label_start_ix", dtype='uint32', data=train_tgt_label_ix)
    f_nmt.create_dataset("train_tgt_label_length", dtype='uint32', data=train_tgt_label_length)
    #f_nmt.create_dataset("train_alignments", dtype='uint32', data=train_alignments)


    f_nmt.create_dataset("valid_src_label", dtype='uint32', data=valid_src_L)
    f_nmt.create_dataset("valid_src_label_start_ix", dtype='uint32', data=valid_src_label_ix)
    f_nmt.create_dataset("valid_src_label_length", dtype='uint32', data=valid_src_label_length)
    f_nmt.create_dataset("valid_tgt_label", dtype='uint32', data=valid_tgt_L)
    f_nmt.create_dataset("valid_tgt_label_start_ix", dtype='uint32', data=valid_tgt_label_ix)
    f_nmt.create_dataset("valid_tgt_label_length", dtype='uint32', data=valid_tgt_label_length)
    #f_nmt.create_dataset("valid_alignments", dtype='uint32', data=valid_alignments)

    f_nmt.close()

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')
    if opt.features_vocabs_prefix:
        saveFeaturesVocabularies('source', dicts['src_features'],
                                 opt.save_data)
        saveFeaturesVocabularies('target', dicts['tgt_features'],
                                 opt.save_data)


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.src_type,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')

    print('Saving data to \'' + opt.save_data + '.alignments.pt\'...')
    save_data = {'train': train['alignments'],
                 'valid': valid['alignments']}
    torch.save(save_data, opt.save_data + '.alignments.pt')

    print('Saving data to \'' + opt.save_data + '.dicts.pt\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.src_type}
    torch.save(save_data, opt.save_data + '.dicts.pt')

def count_aic_langauge():
    imgs = json.load(open('/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/image_captioning/data_chinese.json', 'r'))
    imgs = imgs['images']
    sen_counts = {}
    for img in imgs:
        for sent in img['sentences']:
            sen_counts[str(len(sent['tokens']))] = sen_counts.get(str(len(sent['tokens'])),0) + 1
    cw = sorted([(count, w) for w, count in sen_counts.items()], reverse=True)
    print('top sentences and their counts:')
    print('\n'.join(map(str, cw[:20])))
    item_x = []
    item_count = []
    for _temp in cw:
        item_x.append(int(_temp[1]))
        item_count.append(_temp[0])
    plt.bar(np.asarray(item_x), item_count, 1 / 1.5, color="green")
    plt.show(block=False)

    imgs = json.load(open('/media/jxgu/d4t/dataset/mscoco/caption_datasets/dataset_coco.json', 'r'))
    imgs = imgs['images']
    sen_counts = {}
    for img in imgs:
        for sent in img['sentences']:
            sen_counts[str(len(sent['tokens']))] = sen_counts.get(str(len(sent['tokens'])),0) + 1
    cw = sorted([(count, w) for w, count in sen_counts.items()], reverse=True)
    print('top sentences and their counts:')
    print('\n'.join(map(str, cw[:20])))
    item_x = []
    item_count = []
    for _temp in cw:
        item_x.append(int(_temp[1]))
        item_count.append(_temp[0])
    plt.bar(np.asarray(item_x), item_count, 1 / 1.5, color="green")
    plt.show(block=False)

def clean_en_string(s):
    for c in string.punctuation:
        s = s.replace(c, "")
    return s

def clean_en_punctuations(s):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in s:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def filter_sentences():
    #cont = Contractions('/home/vtt/models/GoogleNews-vectors-negative300.bin')
    #cont.load_models()
    split = 'val'
    if split == 'train':
        src_file = '/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_train_20170912/translation_train_20170912/train.zh'
        tgt_file = '/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_train_20170912/translation_train_20170912/train.en'
        save_src_file = "/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_train_20170912/translation_train_20170912/train_filtered_0303.zh"
        save_tgt_file = "/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_train_20170912/translation_train_20170912/train_filtered_0303.en"
    if split == 'val':
        src_file = '/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en-zh.zh'
        tgt_file = '/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid.en-zh.en'
        save_src_file = "/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid_0303.en-zh.zh"
        save_tgt_file = "/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/ai_challenger_translation_validation_20170912/translation_validation_20170912/valid_0303.en-zh.en"

    srcF_filter = []
    tgtF_filter = []
    srcF = codecs.open(src_file, "r", "utf-8")
    tgtF = codecs.open(tgt_file, "r", "utf-8")
    count = 0
    seen = set()
    while True:
        _sline = srcF.readline()
        _tline = tgtF.readline()

        # normal end of file
        if _sline == "" and _tline == "":
            break

        # source or target does not have same number of lines
        if _sline == "" or _tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = _sline.strip()
        tline = _tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcLine = sline.split()
        tgtLine = tline.split()
        if split == 'train':
            if len(tgtLine)>=5 and len(tgtLine)<=50:
                if _tline not in seen:
                    seen.add(_tline)
                    _sline = re.sub(ur"[%s]+" %punctuation, "", _sline.decode("utf-8"))
                    srcF_filter.append(_sline)
                    #tmp_tline = list(cont.expand_texts([_tline], precise=True))[0]
                    tmp_tline = contractions.fix(_tline)
                    tmp_tline = clean_en_punctuations(clean_en_string(tmp_tline))
                    tgtF_filter.append(tmp_tline.lower())
        else:
            if _tline not in seen:
                seen.add(_tline)
                _sline = re.sub(ur"[%s]+" % punctuation, "", _sline.decode("utf-8"))
                srcF_filter.append(_sline)
                # tmp_tline = list(cont.expand_texts([_tline], precise=True))[0]
                tmp_tline = contractions.fix(_tline)
                tmp_tline = clean_en_punctuations(clean_en_string(tmp_tline))
                tgtF_filter.append(tmp_tline.lower())
        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)
    #random.shuffle(srcF_filter)
    #random.shuffle(tgtF_filter)
    with open(save_src_file, 'w') as f:
        f.writelines(srcF_filter)
    with open(save_tgt_file,'w') as f:
        f.writelines(tgtF_filter)

def extract_coco_captions():
    imgs = json.load(open('/media/jxgu/d4t/dataset/mscoco/caption_datasets/dataset_coco.json', 'r'))
    imgs = imgs['images']
    sents = []
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['sentences'])
        assert n > 0, 'error: some image has no captions'
        for j, s in enumerate(img['sentences']):
            _s = s['raw'].lower()
            sents.append("".join(_s) + "\n")
        counter += n

    with open("/home/jxgu/github/unparied_im2text_jxgu/misc/language-style-transfer/data/mscoco_style.txt",'w+') as f:
        f.writelines(sents)

if __name__ == "__main__":
    #count_aic_langauge()
    #filter_sentences()
    #extract_coco_captions()
    main_h5()