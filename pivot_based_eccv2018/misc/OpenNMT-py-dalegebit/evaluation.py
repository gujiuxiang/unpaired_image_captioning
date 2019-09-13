# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import subprocess
import argparse
#import pyter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pdb

parser = argparse.ArgumentParser(description='evaluate.py')
parser.add_argument('-ref_file', required=True,
                help='Path to reference file')
parser.add_argument('-hyp_file', required=True,
                    help='Path to hypothesis file')
parser.add_argument('-ref_align', required=False,
                help='Path to reference alignments')
parser.add_argument('-hyp_align', required=False,
                    help='Path to hypothesis alignments')

def bleu_score(ref_file, hyp_file):
    """Computes corpus level BLEU score with Moses' multi-bleu.pl script

    Arguments:
        ref_file (str): Path to the reference file
        hyp_file (str): Path to the hypothesis file

    Returns:
        tuple: Tuple (BLEU, details) containing the bleu score and the detailed output of the perl script

    Raises:
        ValueError: Raises error if the perl script fails for some reason
    """
    command = 'perl multi-bleu.pl ' + ref_file + ' < '+hyp_file
    c = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    details, error = c.communicate()
    if not details.startswith('BLEU ='):
        raise ValueError('Error in BLEU score computation:\n' + error)
    else:
        BLEU_str = details.split(' ')[2][:-1]
        BLEU = float(BLEU_str)
        return BLEU, details

def ter_score(ref_file, hyp_file):
    with open(ref_file) as f:
      references = f.readlines()
      references = [ref.strip().split() for ref in references]

    av_ref_len = np.mean([len(ref) for ref in references])

    with open(hyp_file) as f:
      hypothesis = f.readlines()
      hypothesis = [hyp.strip().split() for hyp in hypothesis]

    av_hyp_len = np.mean([len(hyp) for hyp in hypothesis])

    ter_list = [pyter.ter(hyp, ref) for hyp, ref in zip(hypothesis, references)]

    return sum(ter_list)/len(ter_list), av_ref_len, av_hyp_len

def saer_score(ref_align, hyp_align):

    with open(ref_align) as f:
      references = f.readlines()
      references = [ref.strip().split() for ref in references]

    with open(hyp_align) as f:
      hypothesis = f.readlines()
      hypothesis = [hyp.strip().split() for hyp in hypothesis]

    al_prec = 0.0
    al_recall = 0.0
    saer = 0.0
    for al_ref, al_hyp in zip(references, hypothesis):
      al_prec_i = 0
      aer = 0.0
      for al in al_hyp:
        if al in al_ref:
          al_prec_i += 1
      al_count = al_prec_i
      al_prec_i /= len(al_hyp)
      al_rec_i = al_count / len(al_ref)
      al_prec += al_prec_i
      al_recall += al_rec_i
      aer = 1 - (2 * al_count)/(len(al_ref) + len(al_hyp))
      saer += aer
    al_prec /= len(references)
    al_recall /= len(references)
    print("p: %f" % al_prec)
    print("r: %f" % al_recall)

    saer /= len(references)
    return saer
 
def plot_heatmap(model_name, att_weights, idx, srcSent, tgtSent):

    matplotlib.rc('font', **{'sans-serif' : 'Arial',
                           'family' : 'sans-serif'})
    plt.figure(figsize=(20, 18), dpi=80)
    att_weights = np.transpose(att_weights[0][0].cpu().numpy())
    #print("Att_weights:", att_weights.shape)
    #print("Sum-axis-0:", np.sum(att_weights,0))
    plt.imshow(att_weights, cmap='gray', interpolation='nearest')
    srcSent = [str(s) for s in srcSent]
    tgtSent = [str(s) for s in tgtSent]
    plt.xticks(range(0, len(tgtSent)),tgtSent, rotation=45)
    plt.yticks(range(0, len(srcSent)),srcSent)
    plt.savefig("att_" + model_name + "_matrix_" + str(idx) + ".png", bbox_inches='tight')

    plt.close()


def get_fertility(filename, train_filename, src_vocab):

    # list of lists for all sents in training set
    fertility = []

    with open(train_filename) as f:
      sents = f.readlines()
      sents = [line.strip().split() for line in sents]
      sents = [src_vocab.convertToIdx(line, '.') for line in sents]

    with open(filename) as f:
        for i, line in enumerate(f):
            fertility_i = [1] * len(sents[i])
            alignments = line.split(" ")
            for al in alignments:
                idxs = al.split("-")
                a = int(idxs[0])
                b = int(idxs[1])


                fertility_i[a+1] += 1
            #fertility_i = [elem for elem in fertility_i]
            #fertility_i[0] = fertility_i[-1] = 1
            fertility.append(fertility_i)

    return fertility

def get_fert_dict(align_filename, train_filename, src_vocab):
    
    fert_dict = {}
    fertility = []

    with open(train_filename) as f:
      sents = f.readlines()
      sents = [line.strip().split() for line in sents]
      sents = [src_vocab.convertToIdx(line, '.') for line in sents]

    for idx in src_vocab.labelToIdx.values():
      fert_dict[idx] = 1.0

    with open(align_filename) as f:
      lines = f.readlines()
      lines = [line.strip().split() for line in lines]
      for i, line in enumerate(lines):
        fertility_i = [1] * len(sents[i])
        #print(fertility_i)
        #print(line)
        for elem in line:
          idxs = elem.split("-")
          a = int(idxs[0])
          b = int(idxs[1])
          fertility_i[a] += 1
        for idx in sents[i]:
          fert_dict[idx] = max(fert_dict[idx], fertility_i[a])
    return fert_dict      

def getBatchFertilities(fert_dict, batch, default_fert=1.0):
    """
      fert_dict: vocabulary of words and their max fertilities
      batch: src sentences of size(batch_size, src_len)
      returns cudaTensor of size (batch_size, src_len)
    """
    batch_flat = batch.view(-1).data.tolist()
    fertilities = []
    for elem in batch_flat:
        if elem in fert_dict:
            fertilities.append(fert_dict[elem])
        else:
            fertilities.append(default_fert)
    fertilities_tensor = torch.FloatTensor(fertilities).view(batch.size(0), batch.size(1)).cuda()
    return fertilities_tensor
   
     
def main():
    opt = parser.parse_args()
    
    bleu, details = bleu_score(opt.ref_file, opt.hyp_file)
    print("BLEU Score: %f , %s" % (float(bleu), details))
    av_ter , av_ref_len, av_hyp_len = ter_score(opt.ref_file, opt.hyp_file)
    print("TER Score: %f" % av_ter)
    if opt.ref_align:
      saer = saer_score(opt.ref_align, opt.hyp_align)
      print("SAER: %f" % saer)
    print("Average Reference Length: %f" % av_ref_len)
    print("Average Hypothesis Length: %f" % av_hyp_len) 

if __name__ == "__main__":

    main()
