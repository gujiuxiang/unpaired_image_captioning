import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import onmt
import onmt.modules
from onmt.modules import aeq
from onmt.modules.Gate import ContextGateFactory
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math
import numpy as np
import pdb
import evaluation

class Weight_Trans_x(nn.Module):
    def __init__(self, opt):
        super(Weight_Trans_x, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.gen_joint_mask(opt)
        self.get_i2t_wemb()

    def get_i2t_wemb(self):
        # Here we only init the fc model (we use it as the baseline)
        other = torch.load('save/20180222-093200.fc/model-best.pth')
        self.i2t_wemb_weights = other.items()[6][1].cpu()

    def gen_joint_mask(self, opt):
        print('Generate mask for joint vocabulary.')
        load_data = torch.load('data/ai_challenger/machine_translation/nmt_t2t_data_all/nmt_all_0303pivot.joint_vocab.pt')
        self.i2t_pivot_joint_vocab = load_data['i2t_pivot_joint_vocab']
        self.nmt_pivot_joint_vocab = load_data['nmt_pivot_joint_vocab']
        self.i2t_pivot_joint_mask = load_data['i2t_pivot_joint_mask']
        self.nmt_pivot_joint_mask = load_data['nmt_pivot_joint_mask']
        self.maps = torch.Tensor(len(self.i2t_pivot_joint_vocab), 2)
        nmt_pivot_joint_vocab_label2idx = {label: idx for idx, label in self.nmt_pivot_joint_vocab.items()}
        vocab_idx = 0
        for idx, label in self.i2t_pivot_joint_vocab.items():
            if label in nmt_pivot_joint_vocab_label2idx.keys():
                self.maps[vocab_idx,0] = int(idx)
                self.maps[vocab_idx,1] = nmt_pivot_joint_vocab_label2idx[label]
                vocab_idx = vocab_idx + 1
        print('Joint vocabulary for i2t = {} and t2i = {}'.format(len(self.i2t_pivot_joint_vocab), len(self.nmt_pivot_joint_vocab)))

    def mse_loss(self, input, target):
        #return torch.sum((input - target)^2) / input.data.nelement()
        return torch.mean((input - target) ** 2)

    def forward(self, pivot_wemb_nmt):
        _pivot_wemb_i2t = self.i2t_wemb_weights[self.maps[:, 0].long()]
        _pivot_wemb_nmt = pivot_wemb_nmt(Variable(torch.from_numpy(self.maps[:, 1].long().numpy()).cuda(), requires_grad=False))
        loss_0 = self.mse_loss(_pivot_wemb_nmt, Variable(_pivot_wemb_i2t.cuda(), requires_grad=False))
        return loss_0

class Weight_Trans_y(nn.Module):
    def __init__(self, opt):
        super(Weight_Trans_y, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.gen_joint_mask(opt)
        self.get_i2t_wemb()

    def get_i2t_wemb(self):
        # Here we only init the fc model (we use it as the baseline)
        other = torch.load('save/09021117_cnn_resnet101.lm_debug13_scst_.rnn_LSTM/09021117_cnn_resnet101.lm_debug13_scst_.rnn_LSTM.model-best.pth')
        self.i2t_wemb_weights = other.items()[6][1].cpu()

    def gen_joint_mask(self, opt):
        print('Generate mask for joint vocabulary.')
        load_data = torch.load('data/ai_challenger/machine_translation/nmt_t2t_data_all/nmt_all_0303target.joint_vocab.pt')
        self.i2t_target_joint_vocab = load_data['i2t_target_joint_vocab']
        self.nmt_target_joint_vocab = load_data['nmt_target_joint_vocab']
        self.i2t_target_joint_mask = load_data['i2t_target_joint_mask']
        self.nmt_target_joint_mask = load_data['nmt_target_joint_mask']
        self.maps = torch.Tensor(len(self.i2t_target_joint_vocab), 2)
        nmt_target_joint_vocab_label2idx = {label: idx for idx, label in self.nmt_target_joint_vocab.items()}
        vocab_idx = 0
        for idx, label in self.i2t_target_joint_vocab.items():
            if label in nmt_target_joint_vocab_label2idx.keys():
                self.maps[vocab_idx,0] = int(idx)
                self.maps[vocab_idx,1] = nmt_target_joint_vocab_label2idx[label]
                vocab_idx = vocab_idx + 1
        print('Joint vocabulary for i2t = {} and t2i = {}'.format(len(self.i2t_target_joint_vocab), len(self.nmt_target_joint_vocab)))

    def mse_loss(self, input, target):
        #return torch.sum((input - target)^2) / input.data.nelement()
        return torch.mean((input - target) ** 2)

    def forward(self, pivot_wemb_nmt):
        _target_wemb_i2t = self.i2t_wemb_weights[self.maps[:, 0].long()]
        _target_wemb_nmt = pivot_wemb_nmt(Variable(torch.from_numpy(self.maps[:, 1].long().numpy()).cuda(), requires_grad=False))
        loss_0 = self.mse_loss(_target_wemb_nmt, Variable(_target_wemb_i2t.cuda(), requires_grad=False))
        return loss_0

class Embeddings(nn.Module):
    def __init__(self, opt, dicts, feature_dicts=None):
        self.positional_encoding = opt.position_encoding
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(opt.word_vec_size, 5000) \
                .cuda()

        self.word_vec_size = opt.word_vec_size

        super(Embeddings, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        # Word embeddings.
        self.dropout = nn.Dropout(p=opt.dropout)
        self.feature_dicts = feature_dicts
        # Feature embeddings.
        if self.feature_dicts is not None:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(feature_dict.size(),
                             opt.feature_vec_size,
                             padding_idx=onmt.Constants.PAD)
                for feature_dict in feature_dicts])

            # MLP on features and words.
            self.activation = nn.ReLU()
            self.linear = onmt.modules.BottleLinear(
                opt.word_vec_size +
                len(feature_dicts) * opt.feature_vec_size,
                opt.word_vec_size)
        else:
            self.feature_luts = nn.ModuleList([])

    def make_positional_encodings(self, dim, max_len):
        pe = torch.FloatTensor(max_len, 1, dim).fill_(0)
        for i in range(dim):
            for j in range(max_len):
                k = float(j) / (10000.0 ** (2.0 * i / float(dim)))
                pe[j, 0, i] = math.cos(k) if i % 2 == 1 else math.sin(k)
        return pe

    def load_pretrained_vectors(self, emb_file):
        if emb_file is not None:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src_input):
        """
        Embed the words or utilize features and MLP.

        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x input_size
        """
        word = self.word_lut(src_input[:, :, 0])
        emb = word
        if self.feature_dicts is not None:
            features = [feature_lut(src_input[:, :, j + 1])
                        for j, feature_lut in enumerate(self.feature_luts)]

            # Apply one MLP layer.
            emb = self.activation(
                self.linear(torch.cat([word] + features, -1)))

        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)
        return emb


class Encoder(nn.Module):
    """
    Encoder recurrent neural network.
    """

    def __init__(self, opt, dicts, feature_dicts=None):
        """
        Args:
            opt: Model options.
            dicts (`Dict`): The src dictionary
            features_dicts (`[Dict]`): List of src feature dictionaries.
        """
        # Number of rnn layers.
        self.layers = opt.layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, feature_dicts)

        # The Encoder RNN.
        self.encoder_layer = opt.encoder_layer

        if self.encoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerEncoder(self.hidden_size, opt)
                 for i in range(opt.layers)])
        else:
            self.rnn = getattr(nn, opt.rnn_type)(
                input_size, self.hidden_size,
                num_layers=opt.layers,
                dropout=opt.dropout,
                bidirectional=opt.brnn)

        self.fertility = opt.fertility
        self.predict_fertility = opt.predict_fertility
        # opt.supervised_fertility = False
        self.supervised_fertility = opt.supervised_fertility

        self.use_sigmoid_fertility = False  # True
        if self.predict_fertility:
            if self.use_sigmoid_fertility:
                self.fertility_out = nn.Linear(self.hidden_size * self.num_directions + input_size, 1)
            else:
                self.fertility_linear = nn.Linear(self.hidden_size * self.num_directions + input_size,
                                                  2 * self.hidden_size * self.num_directions)
                self.fertility_linear_2 = nn.Linear(2 * self.hidden_size * self.num_directions,
                                                    2 * self.hidden_size * self.num_directions)
                self.fertility_out = nn.Linear(2 * self.hidden_size * self.num_directions, 1, bias=False)
        elif self.supervised_fertility:
            self.sup_linear = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size)
            self.sup_linear_2 = nn.Linear(self.hidden_size, 1, bias=False)

        self.guided_fertility = opt.guided_fertility

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat
            lengths (LongTensor): batch
            hidden: Initial hidden state.

        Returns:
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final
                                    Encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        # CHECKS
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            _, n_batch_ = lengths.size()
            aeq(n_batch, n_batch_)
        # END CHECKS

        emb = self.embeddings(input)
        s_len, n_batch, vec_size = emb.size()

        if self.encoder_layer == "mean":
            # No RNN, just take mean as final state.
            mean = emb.mean(0) \
                .expand(self.layers, n_batch, vec_size)
            return (mean, mean), emb

        elif self.encoder_layer == "transformer":
            # Self-attention tranformer.
            out = emb.transpose(0, 1).contiguous()
            for i in range(self.layers):
                out = self.transformer[i](out, input[:, :, 0].transpose(0, 1))
            return Variable(emb.data), out.transpose(0, 1).contiguous()

        else:
            # import pdb; pdb.set_trace()
            # Standard RNN encoder.
            packed_emb = emb
            if lengths is not None:
                # Lengths data is wrapped inside a Variable.
                lengths = lengths.data.view(-1).tolist()
                packed_emb = pack(emb, lengths)
            outputs, hidden_t = self.rnn(packed_emb, hidden)
            if lengths:
                outputs = unpack(outputs)[0]
            if self.predict_fertility:
                if self.use_sigmoid_fertility:
                    fertility_vals = self.fertility * F.sigmoid(self.fertility_out(
                        torch.cat([outputs.view(-1, self.hidden_size * self.num_directions), emb.view(-1, vec_size)],
                                  dim=1)))
                else:
                    fertility_vals = F.relu(self.fertility_linear(
                        torch.cat([outputs.view(-1, self.hidden_size * self.num_directions), emb.view(-1, vec_size)],
                                  dim=1)))
                    fertility_vals = F.relu(self.fertility_linear_2(fertility_vals))
                    fertility_vals = 1 + torch.exp(self.fertility_out(fertility_vals))
                fertility_vals = fertility_vals.view(n_batch, s_len)
                # fertility_vals = fertility_vals / torch.sum(fertility_vals, 1).repeat(1, s_len) * s_len
            elif self.guided_fertility:
                fertility_vals = None  # evaluation.get_fertility()
            elif self.supervised_fertility:
                fertility_vals = F.relu(self.sup_linear(outputs.view(-1, self.hidden_size * self.num_directions)))
                fertility_vals = F.relu(self.sup_linear_2(fertility_vals))
                fertility_vals = 1 + torch.exp(fertility_vals)
            else:
                fertility_vals = None
            return hidden_t, outputs, fertility_vals


class Decoder(nn.Module):
    """
    Decoder + Attention recurrent neural network.
    """

    def __init__(self, opt, dicts):
        """
        Args:
            opt: model options
            dicts: Target `Dict` object
        """
        self.layers = opt.layers
        self.decoder_layer = opt.decoder_layer
        self._coverage = opt.coverage_attn
        self.exhaustion_loss = opt.exhaustion_loss
        self.fertility_loss = True if opt.supervised_fertility else False
        self.hidden_size = opt.rnn_size
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, None)

        if self.decoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerDecoder(self.hidden_size, opt)
                 for _ in range(opt.layers)])
        else:
            if opt.rnn_type == "LSTM":
                stackedCell = onmt.modules.StackedLSTM
            else:
                stackedCell = onmt.modules.StackedGRU
            self.rnn = stackedCell(opt.layers, input_size,
                                   opt.rnn_size, opt.dropout)
            self.context_gate = None
            if opt.context_gate is not None:
                self.context_gate = ContextGateFactory(
                    opt.context_gate, opt.word_vec_size,
                    input_size, opt.rnn_size, opt.rnn_size
                )

        self.dropout = nn.Dropout(opt.dropout)
        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(
            opt.rnn_size,
            coverage=self._coverage,
            attn_type=opt.attention_type,
            attn_transform=opt.attn_transform,
            c_attn=opt.c_attn
        )
        self.fertility = opt.fertility
        self.predict_fertility = opt.predict_fertility
        self.guided_fertility = opt.guided_fertility
        self.supervised_fertility = opt.supervised_fertility
        # Separate Copy Attention.
        self._copy = False
        if opt.copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                opt.rnn_size, attn_type=opt.attention_type)
            self._copy = True

    def forward(self, input, src, context, state, fertility_vals=None, fert_dict=None, fert_sents=None,
                upper_bounds=None, test=False):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.

        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        t_len, n_batch = input.size()
        s_len, n_batch_, _ = src.size()
        s_len_, n_batch__, _ = context.size()
        aeq(n_batch, n_batch_, n_batch__)

        # aeq(s_len, s_len_)
        # END CHECKS
        if self.decoder_layer == "transformer":
            if state.previous_input:
                input = torch.cat([state.previous_input.squeeze(2), input], 0)
        emb = self.embeddings(input.unsqueeze(2))
        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []

        # Setup the different types of attention.
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        if self.exhaustion_loss:
            attns["upper_bounds"] = []
        if self.fertility_loss:
            attns["predicted_fertility_vals"] = []
            attns["true_fertility_vals"] = []
        if self.decoder_layer == "transformer":
            # Tranformer Decoder.
            assert isinstance(state, TransformerDecoderState)
            output = emb.transpose(0, 1).contiguous()
            src_context = context.transpose(0, 1).contiguous()
            for i in range(self.layers):
                output, attn \
                    = self.transformer[i](output, src_context,
                                          src[:, :, 0].transpose(0, 1),
                                          input.transpose(0, 1))
            outputs = output.transpose(0, 1).contiguous()
            if state.previous_input:
                outputs = outputs[state.previous_input.size(0):]
                attn = attn[:, state.previous_input.size(0):].squeeze()
                attn = torch.stack([attn])
            attns["std"] = attn
            if self._copy:
                attns["copy"] = attn
            state = TransformerDecoderState(input.unsqueeze(2))
        else:
            assert isinstance(state, RNNDecoderState)
            output = state.input_feed.squeeze(0)
            hidden = state.hidden
            # CHECKS
            n_batch_, _ = output.size()
            aeq(n_batch, n_batch_)
            # END CHECKS

            coverage = state.coverage.squeeze(0) \
                if state.coverage is not None else None

            # NOTE: something goes wrong when I try to define a "upper_bounds"
            # variable here -- memory blows up. Apparently the presence of such
            # variable prevents the computation graph to be deleted after
            # processing each batch. I need to investigate this further.
            # A workaround for now is to do one round of softmax (without
            # upper bound constraints) followed by several rounds of constrained
            # softmax.
            # upper_bounds = Variable(torch.ones(attn.size()).cuda())
            # Standard RNN decoder.
            for i, emb_t in enumerate(emb.split(1)):

                # Initialize upper bounds for the current batch

                if upper_bounds is None:
                    # if not test:
                    # 	tgt_lengths = [torch.nonzero(input[:,i].data).size(0) for i in range(n_batch_)]
                    # 	tgt_lengths = torch.Tensor(tgt_lengths).cuda()
                    # else:
                    #    # Maybe the ratio of tgt_len and src_len from training set would be a better estimate
                    #	 tgt_lengths = torch.ones(n_batch_).cuda()
                    if self.predict_fertility:
                        # comp_tensor = torch.Tensor([float(emb.size(0)) / context.size(0)]).repeat(n_batch_, s_len_).cuda()
                        # comp_tensor = (tgt_lengths/s_len_).unsqueeze(1).repeat(1, s_len_).cuda()
                        # print("fertility_vals:", fertility_vals.data)
                        # max_word_coverage = Variable(torch.max(fertility_vals.data, comp_tensor))
                        max_word_coverage = fertility_vals.clone()
                    elif self.guided_fertility:
                        # comp_tensor = torch.Tensor([float(emb.size(0)) / context.size(0)]).repeat(n_batch_, s_len_).cuda()
                        # comp_tensor = (tgt_lengths/s_len_).unsqueeze(1).repeat(1, s_len_).cuda()
                        # import pdb; pdb.set_trace()
                        fertility_vals = Variable(
                            evaluation.getBatchFertilities(fert_dict, src).transpose(1, 0).contiguous())
                        max_word_coverage = fertility_vals
                        # max_word_coverage = Variable(torch.max(fertility_vals, comp_tensor))
                    elif self.supervised_fertility:
                        # k should be index of first sentence in batch
                        predicted_fertility_vals = fertility_vals
                        true_fertility_vals = fert_sents[k: k + n_batch_]
                        if test:
                            max_word_coverage = predicted_fertility_vals
                        else:
                            max_word_coverage = true_fertility_vals
                    else:
                        # max_word_coverage = max(
                        #    self.fertility, float(emb.size(0)) / context.size(0))
                        max_word_coverage = Variable(torch.Tensor([self.fertility]).repeat(n_batch_, s_len_)).cuda()
                        # max_word_coverage = Variable(torch.max(torch.FloatTensor([self.fertility]).repeat(n_batch_).cuda(),
                        #				     tgt_lengths/s_len_).unsqueeze(1).repeat(1, s_len_))
                        #    upper_bounds = -attn + max_word_coverage
                        # else:
                        #    upper_bounds -= attn
                    upper_bounds = max_word_coverage

                # Use <SINK> token for absorbing remaining attention weight

                # import pdb; pdb.set_trace()
                upper_bounds[:, -1] = Variable(100. * torch.ones(upper_bounds.size(0)))
                # if (upper_bounds.size(0) > torch.sum(torch.sum(upper_bounds, 1)).cpu().data.numpy())[0]:
                #    print("inv sum:", torch.sum(upper_bounds, 1))
                #    print("att:", attn)

                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)

                rnn_output, hidden = self.rnn(emb_t, hidden)
                attn_output, attn = self.attn(rnn_output,
                                              context.transpose(0, 1),
                                              upper_bounds=upper_bounds)
                # import pdb; pdb.set_trace()
                # print_attention = True
                # if print_attention:
                #    attn_probs = attn.data.cpu().numpy()
                #    for k in range(attn_probs.shape[0]):
                #        print('\t'.join(str(val) for val in list(attn_probs[k, :])))

                upper_bounds -= attn
                # k_attn = 1
                # upper_bounds = torch.max(upper_bounds - k_attn * attn, Variable(torch.zeros(upper_bounds.size(0), upper_bounds.size(1)).cuda()))
                # if np.any(upper_bounds.cpu().data.numpy()<1):
                #     print("upper bounds less than 1.0")
                # print("attn: ", attn)
                # print("upper_bounds: ", upper_bounds)

                if self.context_gate is not None:
                    output = self.context_gate(
                        emb_t, rnn_output, attn_output
                    )
                    output = self.dropout(output)
                else:
                    output = self.dropout(attn_output)
                outputs += [output]
                attns["std"] += [attn]

                # COVERAGE
                if self._coverage:
                    coverage = (coverage + attn) if coverage else attn
                    attns["coverage"] += [coverage]

                # COPY
                if self._copy:
                    _, copy_attn = self.copy_attn(output,
                                                  context.transpose(0, 1))
                    attns["copy"] += [copy_attn]
                if self.exhaustion_loss:
                    attns["upper_bounds"] += [upper_bounds]
            if self.supervised_fertility:
                attns["true_fertility_vals"] += [true_fertility_vals]
                attns["predicted_fertility_vals"] += [predicted_fertility_vals]
            state = RNNDecoderState(hidden, output.unsqueeze(0),
                                    coverage.unsqueeze(0)
                                    if coverage is not None else None,
                                    upper_bounds)
            outputs = torch.stack(outputs)
            for k in attns:
                attns[k] = torch.stack(attns[k])
        return outputs, state, attns, upper_bounds


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if self.decoder.decoder_layer == "transformer":
            return TransformerDecoderState()
        elif isinstance(enc_hidden, tuple):
            dec = RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:
            dec = RNNDecoderState(self._fix_enc_hidden(enc_hidden))
        dec.init_input_feed(context, self.decoder.hidden_size)
        return dec

    def forward(self, src, tgt, lengths, dec_state=None, fert_dict=None, fert_sents=None):
        """
        Args:
            src, tgt, lengths
            dec_state: A decoder state object

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        # print("src:", src)
        enc_hidden, context, fertility_vals = self.encoder(src, lengths)
        enc_state = self.init_decoder_state(context, enc_hidden)
        out, dec_state, attns, upper_bounds = self.decoder(tgt, src, context,
                                                           enc_state if dec_state is None
                                                           else dec_state, fertility_vals,
                                                           fert_dict, fert_sents)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state, upper_bounds


class DecoderState(object):
    def detach(self):
        for h in self.all:
            if h is not None:
                h.detach_()

    def repeatBeam_(self, beamSize):
        self._resetAll([Variable(e.data.repeat(1, beamSize, 1))
                        for e in self.all])

    def beamUpdate_(self, idx, positions, beamSize):
        for e in self.all:
            a, br, d = e.size()
            sentStates = e.view(a, beamSize, br // beamSize, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, rnnstate, input_feed=None, coverage=None,
                 attn_upper_bounds=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage
        self.attn_upper_bounds = attn_upper_bounds
        self.all = self.hidden + (self.input_feed,)

    def init_input_feed(self, context, rnn_size):
        batch_size = context.size(1)
        h_size = (batch_size, rnn_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        self.all = self.hidden + (self.input_feed,)

    def _resetAll(self, all):
        vars = [Variable(a.data if isinstance(a, Variable) else a,
                         volatile=True) for a in all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
        self.all = self.hidden + (self.input_feed,)

    def beamUpdate_(self, idx, positions, beamSize):
        # I'm overriding this method to handle the upper bounds in the beam
        # updates. May be simpler to add this as part of self.all and not
        # do the overriding.
        # import pdb; pdb.set_trace()
        DecoderState.beamUpdate_(self, idx, positions, beamSize)
        if self.attn_upper_bounds is not None:
            e = self.attn_upper_bounds
            br, d = e.size()
            sentStates = e.view(beamSize, br // beamSize, d)[:, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(0, positions))


class TransformerDecoderState(DecoderState):
    def __init__(self, input=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        self.previous_input = input
        self.all = (self.previous_input,)

    def _resetAll(self, all):
        vars = [(Variable(a.data if isinstance(a, Variable) else a,
                          volatile=True))
                for a in all]
        self.previous_input = vars[0]
        self.all = (self.previous_input,)

    def repeatBeam_(self, beamSize):
        pass
