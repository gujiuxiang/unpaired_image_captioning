from __future__ import division
import onmt.Constants
import numpy as np
import math
import torch
from torch.autograd import Variable

import misc.constants as constants


class onmt_dataset_h5(object):
    """
    Manages dataset creation and usage.

    Example:

        `batch = data[batchnum]`
    """

    def __init__(self, nmt_Data, split, batchSize, cuda,
                 volatile=False, data_type="text",
                 srcFeatures=None, tgtFeatures=None, alignment=None):
        self.src = nmt_Data['train_src_label'] if split == 'train' else nmt_Data['valid_src_label']
        self.src_len = nmt_Data['train_src_label_length'] if split == 'train' else nmt_Data['valid_src_label_length']
        self.srcFeatures = None
        self._type = data_type
        self.tgt = nmt_Data['train_tgt_label'] if split == 'train' else nmt_Data['valid_tgt_label']
        self.tgt_len = nmt_Data['train_tgt_label_length'] if split == 'train' else nmt_Data['valid_tgt_label_length']
        assert (len(self.src) == len(self.tgt))
        self.tgtFeatures = None
        self.cuda = cuda
        self.alignment = alignment
        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, lengths, align_right=False, include_lengths=False, features=None):
        max_length = max(lengths)
        out = torch.Tensor(data.shape[0], max_length.astype(int)).fill_(onmt.Constants.PAD)
        for i in range(data.shape[0]):
            offset = max_length - np.count_nonzero(data[i]) if align_right else 0
            out[i].narrow(0, offset, lengths[i].astype(int)).copy_(torch.from_numpy(data[i,:lengths[i]].astype('int32')))
        return out.long()

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        s = index*self.batchSize
        e = (index+1)*self.batchSize
        batch_size = len(self.src[s:e])
        srclengths = self.src_len[s:e]
        srcBatch = self._batchify(self.src[s:e], srclengths, align_right=False, features=[f[s:e] for f in self.srcFeatures] if self.srcFeatures else None)
        if srcBatch.dim() == 2:
            srcBatch = srcBatch.unsqueeze(2)
        if self.tgt:
            tgtBatch = self._batchify(self.tgt[index*self.batchSize:(index+1)*self.batchSize], self.tgt_len[index*self.batchSize:(index+1)*self.batchSize])
        else:
            tgtBatch = None

        # Create a copying alignment.
        alignment = None
        if self.alignment:
            src_len = srcBatch.size(1)
            tgt_len = tgtBatch.size(1)
            batch = tgtBatch.size(0)
            alignment = torch.ByteTensor(tgt_len, batch, src_len).fill_(0)
            region = self.alignment[s:e]
            for i in range(len(region)):
                alignment[1:region[i].size(1)+1, i,
                          :region[i].size(0)] = region[i].t()
            alignment = alignment.float()

            if self.cuda:
                alignment = alignment.cuda()
        # tgt_len x batch x src_len
        indices = range(len(srcBatch))
        # within batch sorting by decreasing length for variable length rnns
        lengths, perm = torch.sort(torch.from_numpy(srclengths.astype('int32')), 0, descending=True)
        indices = [indices[p] for p in perm]
        srcBatch = [srcBatch[p] for p in perm]
        if tgtBatch is not None:
            tgtBatch = [tgtBatch[p] for p in perm]
        if alignment is not None:
            alignment = alignment.transpose(0, 1)[
                perm.type_as(alignment).long()]
            alignment = alignment.transpose(0, 1).contiguous()

        def wrap(b, dtype="text"):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if dtype == "text":
                b = b.transpose(0, 1).contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # Wrap lengths in a Variable to properly split it in DataParallel
        lengths = lengths.view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        return Batch(wrap(srcBatch, self._type),
                     wrap(tgtBatch, "text"),
                     lengths,
                     indices,
                     batch_size,
                     alignment=alignment)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class Batch(object):
    """
    Object containing a single batch of data points.
    """
    def __init__(self, src, tgt, lengths, indices, batchSize, alignment=None):
        self.src = src
        self.tgt = tgt
        self.lengths = lengths
        self.indices = indices
        self.batchSize = batchSize
        self.alignment = alignment

    def words(self):
        return self.src[:, :, 0]

    def features(self, j):
        return self.src[:, :, j+1]

    def truncate(self, start, end):
        """
        Return a batch containing section from start:end.
        """
        return Batch(self.src, self.tgt[start:end],
                     self.lengths, self.indices, self.batchSize,
                     self.alignment[start:end])
