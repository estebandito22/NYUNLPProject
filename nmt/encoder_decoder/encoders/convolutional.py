from collections import OrderedDict

import numpy as np

import torch
from torch import nn

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
        self.dropout_in = dict_args["dropout_in"]
        self.dropout_out = dict_args["dropout_out"]
        self.model_type = dict_args["model_type"]
        self.attention = dict_args["attention"]

        assert (self.kernel_size == 3) or (self.kernel_size == 5) or (self.kernel_size == 7), \
            "Kernel size must be 3, 5, or 7!"

        self.drop_in = nn.Dropout(p=self.dropout_in)

        self.input_conv = nn.Conv1d(
            in_channels=self.word_embdim, out_channels=self.hidden_size,
            kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        hidden_sizes = [self.hidden_size for _ in range(self.num_layers)]
        self.hidden_conv = nn.Sequential(
            OrderedDict([('l_{}'.format(i // 2),
                          nn.Sequential(
                              nn.Conv1d(in_channels=h_in, out_channels=h_out,
                                        kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2),
                              nn.ReLU(),
                              nn.BatchNorm1d(h_out)))
                         for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.drop_out = nn.Dropout(p=self.dropout_out)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size}
        self.source_word_embd = WordEmbeddings(dict_args)

    def init_hidden(self, batch_size):
        """To work with existing RNN code."""
        return None

    def forward(self, seq_word_indexes, seqlen):
        """Forward pass."""
        # batch_size x seqlen
        batch_size = seq_word_indexes.size(0)
        # seqlen x batch_size x embdim
        seq_word_embds = self.source_word_embd(seq_word_indexes)
        # batch_size x embddim x seqlen
        seq_word_embds = self.drop_in(seq_word_embds).permute(1, 2, 0)

        x = self.input_conv(seq_word_embds)
        self.hidden_conv(x)

        # batch_size x hidden_size x seqlen
        x = self.drop_out(x)

        # seqlen x batch_size x hidden_size
        out = x.permute(2, 0, 1)

        # num_layers x batch_size x hidden_size
        h_n = x.sum(dim=2).unsqueeze(0).expand(self.num_layers, batch_size, -1)
        if self.model_type == 'lstm':
            c_n = h_n
            if self.attention:
                return out, (h_n, c_n)
            else:
                return h_n, (h_n, c_n)

        if self.attention:
            return out, h_n
        return h_n, h_n
