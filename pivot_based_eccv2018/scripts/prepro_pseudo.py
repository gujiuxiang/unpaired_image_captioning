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

def online_eval():
    import time
    import re, string
    from googletrans import Translator
    translator = Translator()
    # Read in the file
    file = open('/home/jxgu/github/im2text_jxgu/pytorch/tmp/flickr_test_1k_zh.txt', 'r')
    zh_lines = file.readlines()
    en_lines = []
    count = 0
    for line in zh_lines:
        #time.sleep(0.1)
        translation = translator.translate(line)
        #en_lines.append(translation.text.replace("there is", ""))
        tmp = translation.text
        #tmp = re.sub('[%s]' % re.escape(string.punctuation), '', tmp)
        en_lines.append(tmp)
        count += 1
        if count % 100 == 0:
            translator = Translator()
            print('... %d sentences prepared' % count)

    with open('tmp/flickr_test_1k_en_google.txt', 'w') as file:
        for line in en_lines:
            file.write("%s\n" % line.encode("utf-8").lower())
    text_id_30K('flickr_test_1k_en_google')
    print('Calculating scores for the generated results ... ...')

    test2json('tmp/flickr_test_1k_en_google')
    lang_stats = language_eval_json_30K('flickr_test_1k_en_google')