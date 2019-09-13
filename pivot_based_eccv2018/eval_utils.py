from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import nltk
import torch
import torch.nn as nn
if float(torch.__version__[0:3]) < 0.4:
    from torch.autograd import Variable
import codecs
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import misc.criterion as criterion
import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
from tqdm import tqdm, trange

def language_eval(type, preds, model_id, split):
    import sys
    if 'coco' in type:
        annFile = 'coco-caption/annotations/captions_val2014.json'
        sys.path.append("coco-caption")
        print("Load reference file from: {}".format(annFile))
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    elif '30k' in type:
        annFile = 'coco-caption/annotations/flickr30k_val.json'
        sys.path.append("coco-caption")
        print("Load reference file from: {}".format(annFile))
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    elif 'zh' in type:
        annFile = 'data/aic_i2t/eval_reference.json'
        sys.path.append("AI_Challenger/Evaluation/caption_eval")
        print("Load reference file from: {}".format(annFile))
        from coco_caption.pycxtools.coco import COCO
        from coco_caption.pycxevalcap.eval import COCOEvalCap
    else:
        raise Exception('Current eval type is not recognizable.')

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', type + '_' + model_id + '_' + split + '.json')
    print("Load cache path is:" + cache_path)
    coco = COCO(annFile)
    valids = coco.getImgIds()
    # filter results to only those in MSCOCO validation set (will be about a third)
    if 'coco' in type:
        preds_filt = [p for p in preds if p['image_id'] in valids]
        print('using %d/%d predictions' % (len(preds_filt), len(preds)))
        json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...
    elif '30k' in type:
        preds_filt = [{'caption': p['caption'], 'image_id': str(p['image_id'])} for p in preds if p['image_id'] in valids]
    else:
        json.dump(preds, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    print(len(set(cocoRes.getImgIds()) & set(coco.getImgIds())))
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    # for p in preds:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def translateBatch(opt, model, batch, src_dict, tgt_dict, beam_accum):
    beamSize = opt.beam_size
    batchSize = batch.batchSize

    #  (1) run the encoder on the src
    encStates, context, fertility_vals = model.encoder(batch.src)
    encStates = model.init_decoder_state(context, encStates)
    if fertility_vals is not None:
        fertility_vals = fertility_vals.repeat(beamSize * batchSize, 1)

    decoder = model.decoder
    attentionLayer = decoder.attn
    useMasking =True

    #  This mask is applied to the attention model inside the decoder
    #  so that the attention ignores source padding
    padMask = None
    if useMasking:
        padMask = batch.words().data.eq(onmt.Constants.PAD).t()

    def mask(padMask):
        if useMasking:
            attentionLayer.applyMask(padMask)

    # (2) if a target is specified, compute the 'goldScore'
    #  (i.e. log likelihood) of the target under the model
    goldScores = context.data.new(batchSize).zero_()

    # (3) run the decoder to generate sentences, using beam search
    # Each hypothesis in the beam uses the same context
    # and initial decoder state
    context = Variable(context.data.repeat(1, beamSize, 1))
    batch_src = Variable(batch.src.data.repeat(1, beamSize, 1))
    decStates = encStates
    decStates.repeatBeam_(beamSize)
    beam = [onmt.Beam(beamSize, True)
            for _ in range(batchSize)]
    if useMasking:
        padMask = batch.src.data[:, :, 0].eq(
            onmt.Constants.PAD).t() \
            .unsqueeze(0) \
            .repeat(beamSize, 1, 1)

    # (3b) The main loop
    upper_bounds = None
    max_sent_length = 100
    for i in range(max_sent_length):
        # (a) Run RNN decoder forward one step.
        mask(padMask)
        input = torch.stack([b.getCurrentState() for b in beam]) \
            .t().contiguous().view(1, -1)
        input = Variable(input, volatile=True)
        decOut, decStates, attn, upper_bounds = model.decoder(input, batch_src,
                                                                   context, decStates,
                                                                   fertility_vals=fertility_vals,
                                                                   fert_dict=None,
                                                                   upper_bounds=decStates.attn_upper_bounds,
                                                                   test=True)

        # import pdb; pdb.set_trace()
        decOut = decOut.squeeze(0)
        # decOut: (beam*batch) x numWords
        attn["std"] = attn["std"].view(beamSize, batchSize, -1).transpose(0, 1).contiguous()

        # (b) Compute a vector of batch*beam word scores.
        out = model.generator.forward(decOut)

        word_scores = out.view(beamSize, batchSize, -1).transpose(0, 1).contiguous()
        # batch x beam x numWords

        # (c) Advance each beam.
        active = []
        for b in range(batchSize):
            is_done = beam[b].advance(word_scores.data[b],
                                      attn["std"].data[b])
            if not is_done:
                active += [b]
            decStates.beamUpdate_(b, beam[b].getCurrentOrigin(),
                                  beamSize)
        if not active:
            break

    # (4) package everything up
    allHyp, allScores, allAttn = [], [], []
    n_best = 1 # If verbose is set, will output the n_best decoded sentences

    for b in range(batchSize):
        scores, ks = beam[b].sortBest()

        allScores += [scores[:n_best]]
        hyps, attn = [], []
        for k in ks[:n_best]:
            hyp, att = beam[b].getHyp(k)
            hyps.append(hyp)
            attn.append(att)
        allHyp += [hyps]
        if useMasking:
            valid_attn = batch.src.data[:, b, 0].ne(onmt.Constants.PAD) \
                .nonzero().squeeze(1)
            attn = [a.index_select(1, valid_attn) for a in attn]
        allAttn += [attn]

        # For debugging visualization.
        if beam_accum:
            beam_accum["beam_parent_ids"].append(
                [t.tolist()
                 for t in beam[b].prevKs])
            beam_accum["scores"].append([["%4f" % s for s in t.tolist()] for t in beam[b].allScores][1:])
            beam_accum["predicted_ids"].append(
                [[tgt_dict.getLabel(id)
                  for id in t.tolist()]
                 for t in beam[b].nextYs][1:])
    # import pdb; pdb.set_trace()
    if fertility_vals is not None:
        cum_attn = allAttn[0][0].sum(0).squeeze(0).cpu().numpy()
        fert = fertility_vals.data[0, :].cpu().numpy()
        for c, f in zip(cum_attn, fert):
            print('%f (%f)' % (c, f))
    # print allAttn[0][0].sum(0)
    return allHyp, allScores, allAttn, goldScores

def eval_split(opt, loader, i2t_model, nmt_model, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    if opt.coco_eval_flag:
        return eval_split_coco_unpaired(opt, loader, i2t_model, nmt_model, eval_kwargs)

    # Make sure in the evaluation mode
    print('Start evaluate the model ...')
    if opt.i2t_eval_flag:
        i2t_crit = criterion.LanguageModelCriterion(opt)
        i2t_model.eval()

    if opt.nmt_eval_flag:
        nmt_crit = criterion.NMT_loss(opt, nmt_model.generator, criterion.NMTCriterion(loader.nmt_dicts['tgt'].size(), opt), eval=True)
        nmt_model.eval()

    loader.reset_iterator(split)
    beam_accum = {"predicted_ids": [], "beam_parent_ids": [], "scores": [], "log_probs": []}

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    if opt.i2t_eval_flag:
        while True:
            data = loader.get_batch(split)
            n = n + loader.batch_size

            if data.get('labels', None) is not None and verbose_loss:
                # forward the model to get loss
                tmp = [data['fc_feats'], data['attri_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else (Variable(torch.from_numpy(_), volatile=True).cuda() if utils.under_0_4() else torch.from_numpy(_).cuda()) for _ in tmp]
                fc_feats, attri_feats, att_feats, labels, masks, att_masks = tmp
                outputs = i2t_model(fc_feats, attri_feats, att_feats, labels, att_masks)
                loss = i2t_crit(outputs, labels[:,1:], masks[:,1:]).data[0]
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1

            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['attri_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
            tmp = [_ if _ is None else (Variable(torch.from_numpy(_), volatile=True).cuda() if utils.under_0_4() else torch.from_numpy(_).cuda()) for _ in tmp]
            fc_feats, attri_feats, att_feats, att_masks = tmp
            # forward the model to also get generated samples for each image
            seq = i2t_model(fc_feats, attri_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
            #print(seq)
            # Print beam search
            if beam_size > 1 and verbose_beam:
                for i in range(loader.batch_size):
                    print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in i2t_model.done_beams[i]]))
                    print('--' * 10)
            sents = utils.decode_sequence(loader.get_vocab(), seq)
            tgtBatch = []

            for k, sent in enumerate(sents):
                if verbose:
                    print('image %s: ' % (data['infos'][k]['id']), sent.encode('utf8', 'replace'))
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                if eval_kwargs.get('dump_path', 0) == 1:
                    entry['file_name'] = data['infos'][k]['file_path']
                predictions.append(entry)
                if eval_kwargs.get('dump_images', 0) == 1:
                    # dump the raw image to vis/ folder
                    cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                    print(cmd)
                    os.system(cmd)

            # if we wrapped around the split or used up val imgs budget then bail
            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if num_images != -1:
                ix1 = min(ix1, num_images)
            for i in range(n - ix1):
                predictions.pop()

            if verbose:
                print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

            if data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break

        lang_stats = None
        if lang_eval == 1:
            if 'coco' in opt.input_json:
                lang_stats = language_eval('coco', predictions, opt.id, split)
            elif 'chinese' in opt.input_json:
                lang_stats = language_eval('zh', predictions, opt.id, split)
            elif '30k' in opt.input_json:
                lang_stats = language_eval('30k', predictions, opt.id, split)
            else:
                raise Exception('Current eval type is not recognizable.')
    # Switch back to training mode
    if opt.nmt_eval_flag:
        for i in tqdm(range(int(loader.nmt_validData.numBatches))):
            batch = loader.get_batch('val')
            outputs, attn, dec_hidden, _ = nmt_model(batch['nmt'].src, batch['nmt'].tgt, batch['nmt'].lengths)
            batch_loss = nmt_crit(loader, batch['nmt'], outputs, attn)

    if opt.nmt_train_flag: nmt_model.train()
    if opt.i2t_train_flag: i2t_model.train()

    if opt.i2t_eval_flag and opt.nmt_eval_flag:
        return loss_sum/loss_evals, predictions, lang_stats, nmt_crit.total_stats.ppl(), nmt_crit.total_stats.accuracy()
    elif opt.nmt_eval_flag:
        return 0.0, None, None, nmt_crit.total_stats.ppl(), nmt_crit.total_stats.accuracy()
    elif opt.i2t_eval_flag:
        return loss_sum / loss_evals, predictions, lang_stats, 0.0, 0.0

def eval_split_coco_unpaired(opt, loader, coco_loader, i2t_model, nmt_model, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    print('Start evaluate the model ...')
    if opt.i2t_eval_flag:
        i2t_crit = criterion.LanguageModelCriterion(opt)
        i2t_model.eval()

    if opt.nmt_eval_flag:
        stats = onmt.Loss.Statistics()
        nmt_crit = criterion.NMT_loss(opt, nmt_model.generator, criterion.NMTCriterion(loader.nmt_dicts['tgt'].size(), opt), eval=True)
        nmt_model.eval()

    loader.reset_iterator(split)

    im_idx = 0
    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    coco_predictions = []

    dump_text = False
    if dump_text: prediction_txt = open('tmp/coco_test_5k_image_path.txt', 'w')

    coco_loader.reset_iterator(split)
    if opt.i2t_eval_flag:
        while True:
            data = loader.get_batch(split)
            coco_data = coco_loader.get_batch(split)
            n = n + coco_loader.batch_size

            if data.get('labels', None) is not None and verbose_loss:
                # forward the model to get loss
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp
                outputs = i2t_model(fc_feats, att_feats, labels, att_masks)
                loss = i2t_crit(outputs, labels[:,1:], masks[:,1:]).data[0]
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1

            # forward the model to also get generated samples for each image
            # Only leave one feature for each image, in case duplicate sample
            coco_fc_feats = Variable(torch.from_numpy(coco_data['fc_feats'][np.arange(coco_loader.batch_size) * coco_loader.seq_per_img]), volatile=True).cuda()
            coco_att_feats = None
            coco_att_mask = None

            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, att_masks = tmp

            # forward the model to also get generated samples for each image
            coco_seq = i2t_model(coco_fc_feats,  coco_att_feats, coco_att_mask, opt=eval_kwargs, mode='sample')[0]
            seq = i2t_model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0]

            # Print beam search
            if beam_size > 1 and verbose_beam:
                for i in range(coco_loader.batch_size):
                    print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in i2t_model.done_beams[i]]))
                    print('--' * 10)
            coco_sents = utils.decode_sequence(coco_loader.get_vocab(), coco_seq)
            sents = utils.decode_sequence(loader.get_vocab(), seq)

            srcBatch = tgtBatch = []
            for coco_sent in coco_sents:
                srcTokens = coco_sent.split()
                srcBatch += [srcTokens]

            # Translate zh-caption (coco) to en
            predBatch = nmt_model.translate(srcBatch)
            # process
            for b in range(len(predBatch)):
                srcSent = ' '.join(srcBatch[b])
                if nmt_model.tgt_dict.lower:
                    srcSent = srcSent.lower()
                #print('%s; PRED: %s' % (srcSent, " ".join(predBatch[b][0])))
                pred_sent = " ".join(predBatch[b][0])
                pred_sent = pred_sent.replace("'s", "is")
                pred_sent = pred_sent.replace("there is", "")
                pred_sent = pred_sent.replace("there 's", "")
                tmpTokens = pred_sent.split()
                predBatch += [tmpTokens]

            for k, coco_sent in enumerate(coco_sents):
                if verbose:
                    im_idx = im_idx + 1
                    print('{}/image: {} | ZH: {} | EN: {}'.format(im_idx, coco_data['infos'][k]['id'], coco_sent.encode('utf8', 'replace').replace(" ", ""), " ".join(predBatch[k][0])))
                    if dump_text: prediction_txt.write('%s\n' % (coco_data['infos'][k]['file_path']))
                entry = {'image_id': data['infos'][k]['id'], 'caption': sents[k]}
                coco_entry = {'image_id': coco_data['infos'][k]['id'], 'caption': " ".join(predBatch[k][0])}
                predictions.append(entry)
                coco_predictions.append(coco_entry)
            # if we wrapped around the split or used up val imgs budget then bail
            ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if num_images != -1:
                ix1 = min(ix1, num_images)
            for i in range(n - ix1):
                predictions.pop()

            coco_ix0 = coco_data['bounds']['it_pos_now']
            coco_ix1 = coco_data['bounds']['it_max']
            if num_images != -1:
                coco_ix1 = min(coco_ix1, num_images)
            for i in range(n - coco_ix1):
                coco_predictions.pop()

            if coco_data['bounds']['wrapped']:
                break
            if num_images >= 0 and n >= num_images:
                break

        if dump_text: prediction_txt.close()
        tag = 'captions_image_info_karpathy_5k_test_results'
        lang_stats = language_eval('zh', predictions, tag, 'val')
        coco_lang_stats = language_eval('en', coco_predictions, tag, 'val')

    # Switch back to training mode
    if opt.nmt_eval_flag:
        loader.reset_iterator(split)
        for i in tqdm(range(int(loader.nmt_validData.numBatches))):
            batch = loader.get_batch(split)
            outputs, attn, dec_hidden, _ = nmt_model(batch['nmt'].src, batch['nmt'].tgt, batch['nmt'].lengths)
            batch_loss, batch_stats = nmt_crit(loader, batch['nmt'], outputs, attn)
            #stats.update(batch_stats)

    if opt.i2t_train_flag: i2t_model.train()
    if opt.nmt_train_flag: nmt_model.train()

    if opt.i2t_eval_flag and opt.nmt_eval_flag:
        return loss_sum/loss_evals, predictions, coco_predictions, lang_stats, coco_lang_stats, nmt_crit.total_stats.ppl(), nmt_crit.total_stats.accuracy()
    elif opt.i2t_eval_flag:
        return loss_sum / loss_evals, predictions, coco_predictions, lang_stats, coco_lang_stats, 0.0, 0.0


def eval_split_coco_paired(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join(
                    [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' % (entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats