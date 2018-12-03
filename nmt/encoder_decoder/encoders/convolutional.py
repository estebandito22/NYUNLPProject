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

        assert (self.kernel_size == 3) or (self.kernel_size == 5) or (self.kernel_size == 7), \
            "Kernel size must be 3, 5, or 7!"

        self.input_conv = nn.Conv1d(
            in_channels=self.word_embdim, out_channels=self.hidden_size,
            kernel_size=self.kernel_size, padding=self.kernel_size // 2)

        hidden_in = [self.hidden_size // 2 for _ in range(self.num_layers)]
        hidden_out = [self.hidden_size for _ in range(self.num_layers)]
        self.convolutions = nn.ModuleList(
            [self.input_conv] + \
            [nn.Conv1d(in_channels=h_in, out_channels=h_out,
                       kernel_size=self.kernel_size,
                       padding=self.kernel_size // 2)
             for (h_in, h_out) in zip(hidden_in, hidden_out)])

        self.drop = nn.Dropout(p=self.dropout)
        self.drop_in = nn.Dropout(p=self.dropout_in)
        self.drop_out = nn.Dropout(p=self.dropout_out)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size}
        self.source_word_embd = WordEmbeddings(dict_args)

        # positional embedding
        self.positional_embedding = nn.Embedding(
            self.max_sent_len, self.word_embdim, self.pad_idx)

        # projections
        # this is different from original paper because we want to fit into
        # existing framework.
        self.proj_word_emb = nn.Linear(self.word_embdim, self.hidden_size)
        self.proj_hidden = nn.Linear(hidden_in[-1], self.hidden_size)
        # residual projections
        self.projections = nn.ModuleList()
        layer_in_channel = [self.word_embdim]
        for out_channels in hidden_in:
            residual = layer_in_channel[-1]
            self.projections.append(nn.Linear(residual, out_channels))
            layer_in_channel.append(out_channels)

    def init_hidden(self, batch_size):
        """To work with existing RNN code."""
        return None

    def forward(self, seq_word_indexes, seqlen):
        """Forward pass."""
        # batch_size x seqlen
        batch_size = seq_word_indexes.size(0)

        # positional embeddings: batch_size x seqlen x embdim
        encoder_padding_mask = seq_word_indexes.eq(self.pad_idx)
        seq_word_positions = torch.zeros_like(seq_word_indexes)
        for i in range(batch_size):
            seq_word_positions[i] = torch.arange(0, self.max_sent_len)
        seq_word_positions.masked_fill_(encoder_padding_mask, self.pad_idx)
        seq_word_positions_embds = self.positional_embedding(seq_word_positions)

        # word embeddings
        # seqlen x batch_size x embdim
        seq_word_embds = self.source_word_embd(seq_word_indexes)

        # combined embeddings
        seq_all_embds = seq_word_embds + seq_word_positions_embds.permute(1, 0, 2)
        # batch_size x embddim x seqlen
        x = self.drop_in(seq_all_embds).permute(1, 2, 0)
        # save input embedding
        input_embedding = x

        # conv layers
        residuals = [x]
        for proj, conv in zip(self.projections, self.convolutions):
            residual = proj(residuals[-1].permute(0, 2, 1)).permute(0, 2, 1)
            x = x.masked_fill(encoder_padding_mask.unsqueeze(1), 0)
            x = self.drop(x)
            x = conv(x)
            x = F.glu(x, dim=1)
            x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # apply mask again
        x = x.masked_fill(encoder_padding_mask.unsqueeze(1), 0)
        x = self.proj_hidden(x.permute(0, 2, 1)).permute(0, 2, 1)

        # project input embedding to hidden size
        # batch size x hidden size x seqlen
        y = self.proj_word_emb(input_embedding.permute(0, 2, 1)).permute(0, 2, 1)

        # output hidden: batch_size x hidden size x seqlen
        out = (x + y) * math.sqrt(0.5)

        # seqlen x batch_size x hidden_size
        out = x.permute(2, 0, 1)

        # num_layers x batch_size x hidden_size
        h_n = out.sum(dim=0).unsqueeze(0).expand(self.num_layers, batch_size, -1)
        if self.model_type == 'lstm':
            c_n = h_n
            if self.attention:
                return out, (h_n, c_n)
            return h_n, (h_n, c_n)

        if self.attention:
            return out, h_n
        return h_n, h_n
