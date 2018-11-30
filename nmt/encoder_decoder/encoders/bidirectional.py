"""PyTorch classes for a recurrent network encoder."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class BidirectionalEncoder(nn.Module):

    """Bidirectional recurrent network to encode sentence."""

    def __init__(self, dict_args):
        """
        Initialize BidirectionalEncoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(BidirectionalEncoder, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]
        self.model_type = dict_args["model_type"]

        # GRU
        if self.model_type == 'gru':
            self.hidden = None
            self.init_hidden(self.batch_size)

            self.rnn = nn.GRU(
                input_size=self.word_embdim, hidden_size=self.hidden_size,
                num_layers=self.num_layers, dropout=self.dropout,
                bidirectional=True)
        elif self.model_type == 'lstm':
            self.hidden = (None, None)
            self.init_hidden(self.batch_size)

            self.rnn = nn.LSTM(
                input_size=self.word_embdim, hidden_size=self.hidden_size,
                num_layers=self.num_layers, dropout=self.dropout,
                bidirectional=True)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size}
        self.source_word_embd = WordEmbeddings(dict_args)

        # initialize weights
        # following https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf
        for name, param in self.rnn.named_parameters():
            if name.find("weight") > -1:
                nn.init.uniform_(param, -0.1, 0.1)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""

        if self.model_type == 'gru':
            hidden = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
            self.hidden = hidden

        elif self.model_type == 'lstm':
            hidden1 = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size)
            hidden2 = torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden1 = hidden1.cuda()
                hidden2 = hidden2.cuda()
            self.hidden = (hidden1, hidden2)

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        if self.model_type == 'gru':
            hidden = self.hidden
        elif self.model_type == 'lstm':
            hidden, c_t = self.hidden
        _, hidden_batch_size, _ = hidden.size()

        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            if self.model_type == 'gru':
                detached_hidden = hidden.detach()
                detached_hidden.zero_()
                self.hidden = detached_hidden
            elif self.model_type == 'lstm':
                detached_c_t = c_t.detach()
                detached_c_t.zero_()
                detached_hidden = hidden.detach()
                detached_hidden.zero_()
                self.hidden = (detached_hidden, detached_c_t)

    def forward(self, seq_word_indexes, seq_lengths):
        """Forward pass."""
        # seqlen x batch x embedding dim
        seq_word_embds = self.source_word_embd(seq_word_indexes)

        seqlen, batch_size, _ = seq_word_embds.size()
        seq_lengths, orig2sorted = seq_lengths.sort(0, descending=True)
        _, sorted2orig = orig2sorted.sort(0, descending=False)
        seq_word_embds = seq_word_embds[:, orig2sorted, :]
        seq_word_embds = pack_padded_sequence(seq_word_embds, seq_lengths)

        if self.model_type == 'gru':
            out, h_n = self.rnn(seq_word_embds, self.hidden)
        elif self.model_type == 'lstm':
            out, (h_n, _) = self.rnn(seq_word_embds, self.hidden)

        out = pad_packed_sequence(out, total_length=seqlen)
        # seqlen x batch size x num_directions * hidden size
        out = out[0][:, sorted2orig, :]

        # numlayers * num directions x batch size x hidden size
        h_n = h_n[:, sorted2orig, :]
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)
        # batch size x hidden size + numlayers * num directions

        return out, h_n
