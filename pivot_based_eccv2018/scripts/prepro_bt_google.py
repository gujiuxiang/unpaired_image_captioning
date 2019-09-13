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

def bt_online():
    from googletrans import Translator
    from time import sleep

    # Read in the file
    file = open('/home/jxgu/github/unparied_im2text_jxgu/data/mscoco/output_cocotalk_sents_60k.txt', 'r')
    en_lines = file.readlines()
    zh_lines = []
    count = 0
    translator = Translator()
    for line in en_lines:
        translation = translator.translate(line, src='en', dest='chinese (simplified)')
        zh_lines.append(translation.text)
        count += 1
        if count % 100 == 0:
            translator = Translator()
            print('... %d sentences prepared' % count)

    tmp_name ='/home/jxgu/github/unparied_im2text_jxgu/data/mscoco/output_cocotalk_sents_60k_zh.txt'
    with open(tmp_name, 'w') as file:
        for line in zh_lines:
            file.write("%s\n" % line.encode("utf-8"))

if __name__ == "__main__":
    bt_online()
