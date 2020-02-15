#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

from __future__ import division

import argparse
import logging
import math
import os

import editdistance

import chainer
import numpy as np
import six
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.scorers.ctc import CTCPrefixScorer

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report_senone(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        reporter.report({'senone_loss_ctc': loss_ctc}, self)
        reporter.report({'senone_loss_att': loss_att}, self)
        reporter.report({'senone_acc': acc}, self)
        reporter.report({'senone_cer_ctc': cer_ctc}, self)
        reporter.report({'senone_cer': cer}, self)
        reporter.report({'senone_wer': wer}, self)
        logging.info('senone_mtl loss:' + str(mtl_loss))
        reporter.report({'senone_loss': mtl_loss}, self)

    def report_phn(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        reporter.report({'phn_loss_ctc': loss_ctc}, self)
        reporter.report({'phn_loss_att': loss_att}, self)
        reporter.report({'phn_acc': acc}, self)
        reporter.report({'phn_cer_ctc': cer_ctc}, self)
        reporter.report({'phn_cer': cer}, self)
        reporter.report({'phn_wer': wer}, self)
        logging.info('phn_mtl loss:' + str(mtl_loss))
        reporter.report({'phn_loss': mtl_loss}, self)

    def report_char(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        reporter.report({'char_loss_ctc': loss_ctc}, self)
        reporter.report({'char_loss_att': loss_att}, self)
        reporter.report({'char_acc': acc}, self)
        reporter.report({'char_cer_ctc': cer_ctc}, self)
        reporter.report({'char_cer': cer}, self)
        reporter.report({'char_wer': wer}, self)
        logging.info('char_mtl loss:' + str(mtl_loss))
        reporter.report({'char_loss': mtl_loss}, self)

    def report_all(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer_ctc': cer_ctc}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        # encoder
        group.add_argument('--etype', default='blstmp', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,
                           help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                                'every y frame at 2nd layer etc.')
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        # attention
        group.add_argument('--atype', default='dot', type=str,
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture')
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--awin', default=5, type=int,
                           help='Window size for location2d attention')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--aconv-chans', default=-1, type=int,
                           help='Number of attention convolution channels \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--aconv-filts', default=100, type=int,
                           help='Number of attention convolution filters \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--sampling-probability', default=0.0, type=float,
                           help='Ratio of predicted labels fed back to decoder')
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?',
                           choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        return parser

    def __init__(self, idim, odims, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)

        self.odim0 = odims[0]
        self.odim1 = odims[1]
        self.odim2 = odims[2]

        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.senone_list = getattr(args, "senone_list", None)
        args.phn_list = getattr(args, "phn_list", None)
        args.char_list = getattr(args, "char_list", None)

        self.senone_list = args.senone_list
        self.phn_list = args.phn_list
        self.char_list = args.char_list

        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()
        self.tap_senone_enc = int(args.tap_enc_senone)
        self.tap_phn_enc = int(args.tap_enc_phn)

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos0 = self.odim0 - 1
        self.sos1 = self.odim1 - 1
        self.sos2 = self.odim2 - 1

        self.eos0 = self.odim0 - 1
        self.eos1 = self.odim1 - 1
        self.eos2 = self.odim2 - 1

        # subsample info
        self.subsample = get_subsample(args, mode='asr', arch='rnn')

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            # Relative importing because of using python3 syntax
            from espnet.nets.pytorch_backend.frontends.feature_transform \
                import feature_transform_for
            from espnet.nets.pytorch_backend.frontends.frontend \
                import frontend_for

            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None

        # encoder
        self.enc = encoder_for(args, idim, self.subsample)
        # ctc
        self.ctc0 = ctc_for(args, self.odim0)
        self.ctc1 = ctc_for(args, self.odim1)
        self.ctc2 = ctc_for(args, self.odim2)
        # attention
        self.att0 = att_for(args)
        self.att1 = att_for(args)
        self.att2 = att_for(args)
        # decoder
        self.dec0 = decoder_for(args, self.odim0, self.sos0, self.eos0, self.att0, labeldist, self.senone_list)
        self.dec1 = decoder_for(args, self.odim1, self.sos1, self.eos1, self.att1, labeldist, self.phn_list)
        self.dec2 = decoder_for(args, self.odim2, self.sos2, self.eos2, self.att2, labeldist, self.char_list)

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        
        self.dec0.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec0.decoder)):
            set_forget_bias_to_one(self.dec0.decoder[l].bias_ih)
        
        self.dec1.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec1.decoder)):
            set_forget_bias_to_one(self.dec1.decoder[l].bias_ih)
        
        self.dec2.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec2.decoder)):
            set_forget_bias_to_one(self.dec2.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pads):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 0. Frontend
        self.combined_loss = 0

        ys_pad_senone = ys_pads[0]
        if self.frontend is not None:
            hs_pad_orig, hlens_orig, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad_orig, hlens_orig = self.feature_transform(hs_pad_orig, hlens_orig)
        else:
            hs_pad_orig, hlens_orig = xs_pad, ilens

        # 1. Encoder
        hs_pad_senone, hlens_senone, _ = self.enc(hs_pad_orig, hlens_orig, tap_layer=self.tap_senone_enc)

        # 2. CTC loss
        if self.mtlalpha == 0:
            self.loss_ctc_senone = None
        else:
            self.loss_ctc_senone = self.ctc0(hs_pad_senone, hlens_senone, ys_pad_senone)

        # 3. attention loss
        if self.mtlalpha == 1:
            self.loss_att_senone, acc = None, None
        else:
            self.loss_att_senone, acc, _ = self.dec0(hs_pad_senone, hlens_senone, ys_pad_senone)
        self.acc_senone = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0 or self.senone_list is None:
            cer_ctc = None
        else:
            cers = []

            y_hats = self.ctc0.argmax(hs_pad_senone).data
            for i, y in enumerate(y_hats):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad_senone[i]

                seq_hat = [self.senone_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.senone_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.blank, '')
                seq_true_text = "".join(seq_true).replace(self.space, ' ')

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                if len(ref_chars) > 0:
                    cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc0.log_softmax(hs_pad_senone).data
            else:
                lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad_senone, torch.tensor(hlens_senone), lpz,
                self.recog_args, self.senone_list,
                self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad_senone[i]

                seq_hat = [self.senone_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.senone_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_att_senone
            loss_att_data = float(self.loss_att_senone)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = self.loss_ctc_senone
            loss_att_data = None
            loss_ctc_data = float(self.loss_ctc_senone)
        else:
            self.loss = alpha * self.loss_ctc_senone + (1 - alpha) * self.loss_att_senone
            loss_att_data = float(self.loss_att_senone)
            loss_ctc_data = float(self.loss_ctc_senone)

        loss_data = float(self.loss)

        senone_loss_ctc_data = loss_ctc_data
        senone_loss_att_data = loss_att_data
        senone_acc = acc
        senone_cer_ctc = cer_ctc
        senone_cer = cer
        senone_wer = wer
        senone_loss_data = loss_data

        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report_senone(senone_loss_ctc_data, senone_loss_att_data, senone_acc, senone_cer_ctc, senone_cer, senone_wer, senone_loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        
        self.combined_loss += self.loss/3
        logging.info("loss DEBUG: senone loss" + str(senone_loss_data))






        ys_pad_phn = ys_pads[1]

        if self.frontend is not None:
            hs_pad_phoneme, hlens_phn, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad_phoneme, hlens_phn = self.feature_transform(hs_pad_phoneme, hlens_phn)
        else:
            hs_pad_phoneme, hlens_phn = xs_pad, ilens

        # 1. Encoder
        hs_pad_phoneme, hlens_phn, _ = self.enc(hs_pad_phoneme, hlens_phn, tap_layer=self.tap_phn_enc)

        # 2. CTC loss
        if self.mtlalpha == 0:
            self.loss_ctc_phn = None
        else:
            self.loss_ctc_phn = self.ctc1(hs_pad_phoneme, hlens_phn, ys_pad_phn)

        # 3. attention loss
        if self.mtlalpha == 1:
            self.loss_att_phn, acc = None, None
        else:
            self.loss_att_phn, acc, _ = self.dec1(hs_pad_phoneme, hlens_phn, ys_pad_phn)
        self.acc_phn = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0 or self.phn_list is None:
            cer_ctc = None
        else:
            cers = []

            y_hats = self.ctc1.argmax(hs_pad_phoneme).data
            for i, y in enumerate(y_hats):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad_phn[i]

                seq_hat = [self.phn_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.phn_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.blank, '')
                seq_true_text = "".join(seq_true).replace(self.space, ' ')

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                if len(ref_chars) > 0:
                    cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc1.log_softmax(hs_pad_phoneme).data
            else:
                lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad_phoneme, torch.tensor(hlens_phn), lpz,
                self.recog_args, self.phn_list,
                self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad_phn[i]

                seq_hat = [self.phn_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.phn_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_att_phn
            loss_att_data = float(self.loss_att_phn)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = self.loss_ctc_phn
            loss_att_data = None
            loss_ctc_data = float(self.loss_ctc_phn)
        else:
            self.loss = alpha * self.loss_ctc_phn + (1 - alpha) * self.loss_att_phn
            loss_att_data = float(self.loss_att_phn)
            loss_ctc_data = float(self.loss_ctc_phn)

        loss_data = float(self.loss)

        phn_loss_ctc_data = loss_ctc_data
        phn_loss_att_data = loss_att_data
        phn_acc = acc
        phn_cer_ctc = cer_ctc
        phn_cer = cer
        phn_wer = wer
        phn_loss_data = loss_data

        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report_phn(phn_loss_ctc_data, phn_loss_att_data, phn_acc, phn_cer_ctc, phn_cer, phn_wer, phn_loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        
        self.combined_loss += self.loss/3
        logging.info("loss DEBUG: phn loss" + str(phn_loss_data))









        ys_pad_char = ys_pads[2]
        if self.frontend is not None:
            hs_pad_char, hlens_char, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad_char, hlens_char = self.feature_transform(hs_pad_char, hlens_char)
        else:
            hs_pad_char, hlens_char = xs_pad, ilens

        # 1. Encoder
        hs_pad_char, hlens_char, _ = self.enc(hs_pad_char, hlens_char)

        # 2. CTC loss
        if self.mtlalpha == 0:
            self.loss_ctc_char = None
        else:
            self.loss_ctc_char = self.ctc2(hs_pad_char, hlens_char, ys_pad_char)

        # 3. attention loss
        if self.mtlalpha == 1:
            self.loss_att_char, acc = None, None
        else:
            self.loss_att_char, acc, _ = self.dec2(hs_pad_char, hlens_char, ys_pad_char)
        self.acc_char = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0 or self.char_list is None:
            cer_ctc = None
        else:
            cers = []

            y_hats = self.ctc2.argmax(hs_pad_char).data
            for i, y in enumerate(y_hats):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad_char[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.blank, '')
                seq_true_text = "".join(seq_true).replace(self.space, ' ')

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                if len(ref_chars) > 0:
                    cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc2.log_softmax(hs_pad_char).data
            else:
                lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad_char, torch.tensor(hlens_char), lpz,
                self.recog_args, self.char_list,
                self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad_char[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_att_char
            loss_att_data = float(self.loss_att_char)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = self.loss_ctc_char
            loss_att_data = None
            loss_ctc_data = float(self.loss_ctc_char)
        else:
            self.loss = alpha * self.loss_ctc_char + (1 - alpha) * self.loss_att_char
            loss_att_data = float(self.loss_att_char)
            loss_ctc_data = float(self.loss_ctc_char)

        loss_data = float(self.loss)

        char_loss_ctc_data = loss_ctc_data
        char_loss_att_data = loss_att_data
        char_acc = acc
        char_cer_ctc = cer_ctc
        char_cer = cer
        char_wer = wer
        char_loss_data = loss_data

        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report_char(char_loss_ctc_data, char_loss_att_data, char_acc, char_cer_ctc, char_cer, char_wer, char_loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        
        self.reporter.report_all(senone_loss_ctc_data + phn_loss_ctc_data + char_loss_ctc_data, \
            senone_loss_att_data + phn_loss_att_data + char_loss_att_data, \
                (senone_acc + phn_acc + char_acc)/3, \
                    (senone_cer_ctc + phn_cer_ctc + char_cer_ctc)/3, \
                        (senone_cer + phn_cer + char_cer)/3, \
                            (senone_wer + phn_wer + char_wer)/3, \
                                (senone_loss_data + phn_loss_data + char_loss_data)/3)

        self.combined_loss += self.loss/3
        logging.info("loss DEBUG: char loss" + str(char_loss_data))



        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. encoder
        hs, _, _ = self.enc(hs, hlens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, recog_args, char_list, rnnlm)
        return y

    def recognize_batch(self, xs, recog_args, char_list, rnnlm=None, trans_type = "phn"):
        """E2E beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(xs_pad, ilens)
            hs_pad, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        if(trans_type == "senone"):
            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc0.log_softmax(hs_pad)
            else:
                lpz = None
            # 2. decoder
            hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
            y = self.dec0.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, char_list, rnnlm)
        elif(trans_type == "phn"):
            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc1.log_softmax(hs_pad)
            else:
                lpz = None
            # 2. decoder
            hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
            y = self.dec1.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, char_list, rnnlm)
        elif(trans_type == "char"):
            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc2.log_softmax(hs_pad)
            else:
                lpz = None
            # 2. decoder
            hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
            y = self.dec2.recognize_beam_batch(hs_pad, hlens, lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def enhance(self, xs):
        """Forward only in the frontend stage.

        :param ndarray xs: input acoustic feature (T, C, F)
        :return: enhaned feature
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise RuntimeError('Frontend does\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)
        enhanced, hlensm, mask = self.frontend(xs_pad, ilens)
        if prev:
            self.train()
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pads, decoder_id=2):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        ys_pad = ys_pads[decoder_id]
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
                hs_pad, hlens = self.feature_transform(hs_pad, hlens)
            else:
                hs_pad, hlens = xs_pad, ilens

            # 1. Encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # 2. Decoder
            if(decoder_id == 0):
                att_ws = self.dec0.calculate_all_attentions(hpad, hlens, ys_pad)
            elif(decoder_id == 1):
                att_ws = self.dec1.calculate_all_attentions(hpad, hlens, ys_pad)
            elif(decoder_id == 2):
                att_ws = self.dec2.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen
