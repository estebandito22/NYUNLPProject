from collections import OrderedDict

import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class ConvolutionalEncoder(nn.Module):

    def __init__(self, dict_args):
        """
        Initialize BidirectionalEncoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(ConvolutionalEncoder, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.kernel_size = dict_args["kernel_size"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.dropout_in = dict_args["dropout_in"]
        self.dropout_out = dict_args["dropout_out"]
        self.model_type = dict_args["model_type"]
        self.attention = dict_args["attention"]
        self.pad_idx = dict_args["pad_idx"]
        self.max_sent_len = dict_args["max_sent_len"]

        assert (self.kernel_size == 3) \
            or (self.kernel_size == 5) or (self.kernel_size == 7), \
            "Kernel size must be 3, 5, or 7!"

        # context network
        # context encoder has migh have 1/2 hidden dim of attention encoder per
        # https://arxiv.org/pdf/1611.02344.pdf
        hidden_in = [self.word_embdim] \
            + [self.hidden_size for _ in range(self.num_layers - 1)]
        hidden_out = [self.hidden_size for _ in range(self.num_layers)]
        self.convolutions_c = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=h_in, out_channels=h_out,
                                     kernel_size=self.kernel_size,
                                     padding=self.kernel_size // 2), nn.Tanh())
             for (h_in, h_out) in zip(hidden_in, hidden_out)])

        # attention network
        hidden_in = [self.word_embdim] \
            + [self.hidden_size for _ in range(self.num_layers * 3 - 1)]
        hidden_out = [self.hidden_size for _ in range(self.num_layers * 3)]
        self.convolutions_a = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=h_in, out_channels=h_out,
                                     kernel_size=self.kernel_size,
                                     padding=self.kernel_size // 2), nn.Tanh())
             for (h_in, h_out) in zip(hidden_in, hidden_out)])

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size,
                     'kernel_size': self.kernel_size}
        self.source_word_embd = WordEmbeddings(dict_args)

        # positional embedding
        self.positional_embedding = nn.Embedding(
            self.max_sent_len, self.word_embdim, self.pad_idx)

        nn.init.uniform_(self.positional_embedding.weight, -0.05, 0.05)

        # dropout
        self.drop_in = nn.Dropout(p=self.dropout_in)
        self.drop_out = nn.Dropout(p=self.dropout_out)

        # input word projection
        # self.proj_word_emb = nn.Linear(self.word_embdim, self.hidden_size)

        # weight initializations
        for layer in self.convolutions_c:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer[0].weight)
            nn.init.uniform_(layer[0].weight,
                             (self.kernel_size * fan_in)**(-0.5) * -1,
                             (self.kernel_size * fan_in)**(-0.5))

        for layer in self.convolutions_a:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer[0].weight)
            nn.init.uniform_(layer[0].weight,
                             (self.kernel_size * fan_in)**(-0.5) * -1,
                             (self.kernel_size * fan_in)**(-0.5))

        # scale gradients
        for layer in self.convolutions_c:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer[0].weight)
            layer[0].weight.register_hook(
                lambda grad: grad / math.sqrt(fan_in))
            layer[0].bias.register_hook(
                lambda grad: grad / math.sqrt(fan_in))

        for layer in self.convolutions_a:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer[0].weight)
            layer[0].weight.register_hook(
                lambda grad: grad / math.sqrt(fan_in))
            layer[0].bias.register_hook(
                lambda grad: grad / math.sqrt(fan_in))

    def init_hidden(self, batch_size):
        """To work with existing RNN code."""
        return None

    def forward(self, seq_word_indexes, seqlen):
        """Forward pass."""
        # batch_size x seqlen
        batch_size, seqlen = seq_word_indexes.size()

        # positional embeddings: seqlen x batch size  x embdim
        encoder_padding_mask = seq_word_indexes.eq(self.pad_idx)
        seq_word_positions = torch.zeros_like(seq_word_indexes)
        for i in range(batch_size):
            seq_word_positions[i] = torch.arange(0, self.max_sent_len)
        seq_word_positions.masked_fill_(encoder_padding_mask, self.pad_idx)
        seq_word_positions_embds = self.positional_embedding(
            seq_word_positions)
        seq_word_positions_embds = seq_word_positions_embds.permute(1, 0, 2)

        # word embeddings
        # seqlen x batch_size x embdim
        seq_word_embds = self.source_word_embd(seq_word_indexes)

        # combined embeddings
        seq_all_embds = seq_word_embds + seq_word_positions_embds

        # batch size x hidden size x seqlen
        x = self.drop_in(seq_all_embds).permute(1, 2, 0)

        # init residuals
        residual_a = torch.zeros([batch_size, self.hidden_size, seqlen])
        residual_c = torch.zeros([batch_size, self.hidden_size, seqlen])
        if torch.cuda.is_available():
            residual_a = residual_a.cuda()
            residual_c = residual_c.cuda()

        # conv layers
        a = x
        residuals_a = [residual_a]
        for i in range(len(self.convolutions_a)):
            # get all but last convs
            conv_a = self.convolutions_a[i]
            # forward attention network
            residual_a = residuals_a[-1]
            a = conv_a(a)
            a = a + residual_a
            residuals_a.append(a)

        c = x
        residuals_c = [residual_c]
        for i in range(len(self.convolutions_c)):
            # get all but last convs
            conv_c = self.convolutions_c[i]
            # forward context network
            residual_c = residuals_c[-1]
            c = conv_c(c)
            c = c + residual_c
            residuals_c.append(c)

        # -> seqlen x batch size x hidden size
        c = c.permute(2, 0, 1)
        a = a.permute(2, 0, 1)

        # c = self.drop_out(c)

        # batch_size x hidden_size
        h_n = c.sum(dim=0)
        if self.model_type == 'lstm':
            c_n = h_n
            if self.attention:
                return (c, a), (h_n, c_n)
            return h_n, (h_n, c_n)

        if self.attention:
            return (c, a), h_n
        return h_n, h_n
