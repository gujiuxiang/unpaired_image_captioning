import sys
import codecs
import os
import math
import operator
import json



def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' or '.en' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp


def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best


def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp


def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))


def BLEU(candidate, references):
    precisions = []
    pr, bp = count_ngram(candidate, references, 1)
    precisions.append(pr)
    bleu1 = geometric_mean(precisions) * bp

    pr, bp = count_ngram(candidate, references, 2)
    precisions.append(pr)
    bleu2 = geometric_mean(precisions) * bp

    pr, bp = count_ngram(candidate, references, 3)
    precisions.append(pr)
    bleu3 = geometric_mean(precisions) * bp

    pr, bp = count_ngram(candidate, references, 4)
    precisions.append(pr)
    bleu4 = geometric_mean(precisions) * bp
    print(bleu1, bleu2, bleu3, bleu4)
    return [bleu1, bleu2, bleu3, bleu4]

if __name__ == "__main__":
    from googletrans import Translator
    translator = Translator()
    # Read in the file
    '''
    file = open('/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/nmt_t2t_data_all/valid_0303.zh', 'r')
    zh_lines = file.readlines()
    en_lines = []
    count = 0
    for line in zh_lines:
        #translation = line
        translation = translator.translate(line)
        en_lines.append(translation.text)
        count += 1
        if count % 1000 == 0:
            translator = Translator()
            print('... %d sentences prepared' % count)
    
    with open('/home/jxgu/github/unparied_im2text_jxgu/tmp/aic_nmt_val_5k_zh_online.en.txt', 'w') as file:
        for line in en_lines:
            file.write("%s\n" % line.encode("utf-8").lower())
    '''
    candidate, references = fetch_data('/home/jxgu/github/unparied_im2text_jxgu/tmp/aic_nmt_val_5k_zh_online.en.txt', '/home/jxgu/github/unparied_im2text_jxgu/data/ai_challenger/machine_translation/nmt_t2t_data_all/valid_0303.en')
    import difflib
    acc = 0.0
    for i in range(len(candidate)):
        acc = acc + difflib.SequenceMatcher(None, candidate[i], references[0][i]).ratio()
    acc = acc/len(candidate)
    bleu = BLEU(candidate, references)
    print bleu
    out = open('bleu_out.txt', 'w')
    out.write(str(bleu))
    out.close()
