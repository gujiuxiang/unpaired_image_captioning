import onmt
import onmt.Models
import onmt.modules
import onmt.IO
import torch.nn as nn
import torch
from torch.autograd import Variable
import evaluation
import pdb

class nmt_translator(object):
    def __init__(self, opt, src_dict, tgt_dict):
        self.opt = opt
        self.tt = torch.cuda
        self.beam_accum = None

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.align = self.src_dict.align(self.tgt_dict)
        self.src_feature_dicts = None
        self._type = "text"

        self.copy_attn = opt.copy_attn
        self.predict_fertility = False

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def buildData(self, srcBatch, goldBatch):
        srcFeats = []
        srcData = []
        tgtData = []
        for b in srcBatch:
            _, srcD, srcFeat = onmt.IO.readSrcLine(b, self.src_dict,
                                                   self.src_feature_dicts,
                                                   self._type)
            srcData += [srcD]
            for i in range(len(srcFeats)):
                srcFeats[i] += [srcFeat[i]]

        if goldBatch:
            for b in goldBatch:
                _, tgtD, tgtFeat = onmt.IO.readTgtLine(b, self.src_dict,
                                                       None, self._type)
                tgtData += [tgtD]

        return onmt.Dataset(srcData, tgtData, self.opt.batch_size,
                            True, volatile=True,
                            data_type=self._type,
                            srcFeatures=srcFeats)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        for i in range(len(tokens)):
            if tokens[i] == onmt.Constants.UNK_WORD:
                _, maxIndex = attn[i].max(0)
                tokens[i] = src[maxIndex[0]]

        return tokens

    def translateBatch(self, model, batch):
        beamSize = self.opt.beam_size
        batchSize = batch.batchSize

        #  (1) run the encoder on the src
        encStates, context = model.encoder(batch.src)
        encStates = model.init_decoder_state(context, encStates)

        useMasking = True

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = None
        if useMasking:
            padMask = batch.words().data.eq(onmt.Constants.PAD).t()

        def mask(padMask):
            if useMasking:
                model.decoder.attn.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()

        #  (3) run the decoder to generate sentences, using beam search
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

        #  (3b) The main loop
        upper_bounds = None
        for i in range(100):
            # (a) Run RNN decoder forward one step.
            mask(padMask)
            input = torch.stack([b.getCurrentState() for b in beam])\
                         .t().contiguous().view(1, -1)
            input = Variable(input, volatile=True)
            decOut, decStates, attn, upper_bounds = model.decoder(input, batch_src,
                                                         context, decStates,
                                                         upper_bounds=decStates.attn_upper_bounds,
                                                         test=True)

            #import pdb; pdb.set_trace()
            decOut = decOut.squeeze(0)
            # decOut: (beam*batch) x numWords
            attn["std"] = attn["std"].view(beamSize, batchSize, -1) \
                                     .transpose(0, 1).contiguous()

            # (b) Compute a vector of batch*beam word scores.
            if not self.copy_attn:
                out = model.generator.forward(decOut)
            else:
                # Copy Attention Case
                words = batch.words().t()
                words = torch.stack([words[i] for i, b in enumerate(beam)])\
                             .contiguous()
                attn_copy = attn["copy"].view(beamSize, batchSize, -1) \
                                        .transpose(0, 1).contiguous()

                out, c_attn_t \
                    = model.generator.forward(
                        decOut, attn_copy.view(-1, batch_src.size(0)))

                for b in range(out.size(0)):
                    for c in range(c_attn_t.size(1)):
                        v = self.align[words[0, c].data[0]]
                        if v != onmt.Constants.PAD:
                            out[b, v] += c_attn_t[b, c]
                out = out.log()

            word_scores = out.view(beamSize, batchSize, -1) \
                .transpose(0, 1).contiguous()
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

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        self.n_best = 1

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:self.n_best]]
            hyps, attn = [], []
            for k in ks[:self.n_best]:
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
            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])
        #print allAttn[0][0].sum(0)
        return allHyp, allScores, allAttn, goldScores

    def translate(self, model, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset[0]
        batchSize = batch.batchSize

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(model, batch)
        pred, predScore, attn, goldScore = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore, batch.indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.n_best)]
            )

        return predBatch, predScore, goldScore, attn, batch.src