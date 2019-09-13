from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import nltk
from nltk.translate.bleu_score import SmoothingFunction
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import time
import math
import sys
import os
import json
from multiprocessing import Pool

def under_0_4():
    return float(torch.__version__[0:3]) < 0.4

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

class create_args(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

def transfer_args(args_src):
    args_tgt = {}
    for _arg in args_src._get_kwargs():
        if 'nmt_' == _arg[0][:4]:
            args_tgt[_arg[0][4:]] = _arg[1]
    return create_args(args_tgt)

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                if float(torch.__version__[0:3]) < 0.4:
                    txt = txt + ix_to_word[str(ix)]
                else:
                    txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (name, scoreTotal / wordsTotal, name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def calc_bleu(reference, hypothesis, weight):
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1)

def self_bleu(reference, ngram):
    #ngram = ngram
    weight = tuple((1. / ngram for _ in range(ngram)))
    pool = Pool(os.cpu_count())
    result = list()
    sentence_num = len(reference)
    for index in range(sentence_num):
        hypothesis = reference[index]
        other = reference[:index] + reference[index+1:]
        result.append(pool.apply_async(calc_bleu, args=(other, hypothesis, weight)))

    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def cocojson2text(zh_json_file, zh_text_file):
    f = open(zh_text_file, 'w')
    test = json.load(open(zh_json_file, 'r'))
    for cap in test:
        f.write(cap['caption'].encode('utf8')+'\n')
    f.close()

def get_reference_all(test_data):
    reference = list()
    with open(test_data) as real_data:
        for text in real_data:
            text = nltk.word_tokenize(text)
            reference.append(text)
    reference = reference
    return reference

def get_reference_single(test_data):
    reference = list()
    file = open(test_data, 'r')
    gt_lines = file.readlines()
    for i in range(len(gt_lines)):
        if i % 5==0:
            text = gt_lines[i]
            text = nltk.word_tokenize(text)
            reference.append(text)
    reference = reference
    return reference

def text2textid(en_text_file, en_json_file):
    test = json.load(open(en_json_file, 'r'))
    image_id = []
    for cap in test:
        image_id.append(cap['image_id'])
    image_captions = []
    with open(en_text_file + '.txt', 'r') as opfd:
        for line in opfd:
            image_captions.append(line.strip().split('\t')[0])

    #prediction_txt = open(os.path.splitext(en_text_file)[0] + '_id.txt', 'w')
    prediction_txt = open(en_text_file + '_id.txt', 'w')
    for i in range(len(image_id)):
        prediction_txt.write('%d\t%s\n' % (image_id[i], image_captions[i]))
    print('Writing predictions to file ... ...')
    prediction_txt.close()

def visulize_dep(sentences):
    # sentences = ["This is an example.", "This is another one."]
    import spacy
    from spacy import displacy
    from pathlib import Path
    nlp = spacy.load('en')
    for sent in sentences:
        doc = nlp(sent)
        svg = displacy.render(doc, style='dep')
        file_name = '-'.join([w.text for w in doc if not w.is_punct]) + '.svg'
        output_path = Path('tmp/' + file_name)
        output_path.open('w', encoding='utf-8').write(svg)

class _CocoResFormat:

    def __init__(self):
        self.res = []
        self.caption_dict = {}

    def read_multiple_files(self, filelist, hash_img_name):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename, hash_img_name)

    def read_file(self, filename, hash_img_name):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                if len(id_sent) > 2:
                    id_sent = id_sent[-2:]
                assert len(id_sent) == 2
                sent = id_sent[1]

                if hash_img_name:
                    img_id = int(int(hashlib.sha256(id_sent[0].encode('utf8')).hexdigest(),
                                     16) % sys.maxsize)
                else:
                    img = id_sent[0].split('_')[-1].split('.')[0]
                    img_id = int(img)
                imgid_sent = {}

                if img_id in self.caption_dict:
                    assert self.caption_dict[img_id] == sent
                else:
                    self.caption_dict[img_id] = sent
                    imgid_sent['image_id'] = img_id
                    imgid_sent['caption'] = sent
                    self.res.append(imgid_sent)

    def dump_json(self, outfile):
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))

def test2cocojson(filename):
    sys.path.append('../coco-caption/')
    from pycocotools.coco import COCO
    crf = _CocoResFormat()
    crf.read_file(filename+'.txt', False)
    # Remove it first !!!
    if os.path.exists(filename+'.json'):
        os.remove(filename+'.json')
    crf.dump_json(filename+'.json')


def create_zh_en_html():
    head = '<html>' \
           '<body>' \
           '<h1>Image Captions (zh + en)</h1>' \
           '<table border="1px solid gray" style="width=100%">' \
           '<tr>' \
           '<td><b>Image</b></td>' \
           '<td><b>ID</b></td>' \
           '<td><b>Image ID</b></td>' \
           '<td><b>Generated Caption zh</b></td>' \
           '<td><b>Generated Caption en</b></td>' \
           '</tr>'

    image_paths = []
    # os.system('rm -rf ' + format('tmp/coco_test_5k_image_path.txt'))
    with open('tmp/coco_test_5k_image_path.txt', 'r') as opfd:
        for line in opfd:
            image_paths.append(line.strip().split('\t')[0])
    image_cap_zh = []
    # os.system('rm -rf ' + format('tmp/coco_test_5k_zh_bak.txt'))
    with open('tmp/coco_test_5k_zh_bak.txt', 'r') as opfd:
        for line in opfd:
            image_cap_zh.append(line.strip().split('\t')[0])
    image_cap_en = []
    # os.system('rm -rf ' + format('tmp/coco_test_5k_en.txt'))
    with open('tmp/coco_test_5k_zh.en.txt', 'r') as opfd:
        for line in opfd:
            image_cap_en.append(line.strip().split('\t')[0])

    fname_html = 'eval_results/caption_zh_en.html'
    os.system('rm -rf ' + format(fname_html))
    os.system('echo ' + '"' + head + '"' + ' >> ' + fname_html)
    for i in range(5000):
        dump_line = '\n<tr><td><img src={} width="100"><td>{}</td></td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            '/home/jxgu/github/visual_word_embedding/order_emb/data/mscoco/' + image_paths[i], i, image_paths[i], image_cap_zh[i], image_cap_en[i])
        os.system('echo ' + '"' + dump_line + '"' + ' >> ' + fname_html)

def filter(injson, intext):
    target_vocab =json.load(open('/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/image_captioning/chinese_talk.json'))
    tgtF_filter = []
    tgtF = codecs.open('tmp/coco_test_5k_en.txt', "r", "utf-8")
    count = 0
    while True:
        _tline = tgtF.readline()
        tline = _tline.strip()
        tgtLine = tline.split()
        tgtF_filter.append(_tline)
        count += 1

        if count % 100 == 0:
            print('... %d sentences prepared' % count)

    #text_id()
    #print('Calculating scores for the generated results ... ...')
    #test2json()
    #lang_stats = language_eval()


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr': self.current_lr,
                'scheduler_state_dict': {key: value for key, value in self.scheduler.__dict__.items() if key not in {'optimizer', 'is_better'}},
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr)  # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.__dict__.update(state_dict['scheduler_state_dict'])
            self.scheduler._init_is_better(mode=self.scheduler.mode, threshold=self.scheduler.threshold, threshold_mode=self.scheduler.threshold_mode)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def get_std_opt(model, factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt(model.model.tgt_embed[0].d_model, factor, warmup,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
