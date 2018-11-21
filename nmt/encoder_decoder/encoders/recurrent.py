"""PyTorch classes for a recurrent network encoder."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class RecurrentEncoder(nn.Module):

    """Recurrent network to encode sentence."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentEncoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(RecurrentEncoder, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]

        # GRU
        self.hidden = None
        self.init_hidden(self.batch_size)

        self.rnn = nn.GRU(
            input_size=self.word_embdim, hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
            bidirectional=False)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size}
        self.source_word_embd = WordEmbeddings(dict_args)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        if torch.cuda.is_available():
            self.hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size).cuda()
        else:
            self.hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size)

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        _, hidden_batch_size, _ = self.hidden.size()
        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            detached_hidden = self.hidden.detach()
            detached_hidden.zero_()
            self.hidden = detached_hidden

    def forward(self, seq_word_indexes, seq_lengths):
        """Forward pass."""
        seq_word_embds = self.source_word_embd(seq_word_indexes)

        _, batch_size, _ = seq_word_embds.size()
        seq_lengths, orig2sorted = seq_lengths.sort(0, descending=True)
        _, sorted2orig = orig2sorted.sort(0, descending=False)
        seq_word_embds = seq_word_embds[:, orig2sorted, :]
        seq_word_embds = pack_padded_sequence(seq_word_embds, seq_lengths)

        _, out = self.rnn(seq_word_embds, self.hidden)
        out = out[:, sorted2orig, :]
        out = out.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return out, out
