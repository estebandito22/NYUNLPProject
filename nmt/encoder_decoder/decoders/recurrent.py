"""PyTorch class for a recurrent network sentence decoder."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nmt.encoder_decoder.decoders.attention import AttentionMechanism
from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class RecurrentDecoder(nn.Module):

    """Recurrent Decoder to decode encoded sentence."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentDecoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(RecurrentDecoder, self).__init__()
        self.enc_hidden_dim = dict_args["enc_hidden_dim"]
        self.enc_num_layers = dict_args["enc_num_layers"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.hidden_size = dict_args["hidden_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.model_type = dict_args["model_type"]
        # print(self.model_type)

        # attention
        if self.attention:

            if self.model_type == 'gru':
                self.hidden = None
                # self.init_hidden_zeros(self.batch_size)
                self.rnn = nn.GRU(
                    self.enc_hidden_dim * 2 + self.word_embdim,
                    self.hidden_size, num_layers=self.num_layers,
                    dropout=self.dropout)

                self.init_hidden = nn.Linear(
                    self.enc_num_layers * self.enc_hidden_dim * 2,
                    self.hidden_size * self.num_layers)

            elif self.model_type == 'lstm':
                self.hidden = None
                # self.init_hidden_zeros(self.batch_size)
                self.rnn = nn.LSTM(
                    self.enc_hidden_dim * 2 + self.word_embdim,
                    self.hidden_size, num_layers=self.num_layers,
                    dropout=self.dropout)

                self.init_hidden1 = nn.Linear(
                    self.enc_num_layers * self.enc_hidden_dim * 2,
                    self.hidden_size * self.num_layers)

                self.init_hidden2 = nn.Linear(
                    self.enc_num_layers * self.enc_hidden_dim * 2,
                    self.hidden_size * self.num_layers)

            dict_args = {'context_size': self.max_sent_len,
                         'context_dim': self.enc_hidden_dim * 2,
                         'hidden_size': self.hidden_size * self.num_layers}
            self.attn_layer = AttentionMechanism(dict_args)

        else:
            if self.model_type == 'gru':
                self.hidden = None
                # self.init_hidden_zeros(self.batch_size)
                self.rnn = nn.GRU(
                    self.enc_num_layers * self.enc_hidden_dim + self.word_embdim,
                    self.hidden_size, num_layers=self.num_layers,
                    dropout=self.dropout)

                self.init_hidden = nn.Linear(
                    self.enc_num_layers * self.enc_hidden_dim,
                    self.hidden_size * self.num_layers)

            elif self.model_type == 'lstm':
                self.hidden = None
                # self.init_hidden_zeros(self.batch_size)
                self.rnn = nn.LSTM(
                    self.enc_num_layers * self.enc_hidden_dim + self.word_embdim,
                    self.hidden_size, num_layers=self.num_layers,
                    dropout=self.dropout)

                self.init_hidden1 = nn.Linear(
                    self.enc_num_layers * self.enc_hidden_dim,
                    self.hidden_size * self.num_layers)

                self.init_hidden2 = nn.Linear(
                    self.enc_num_layers * self.enc_hidden_dim,
                    self.hidden_size * self.num_layers)

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

        # target embeddings
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size}
        self.target_word_embd = WordEmbeddings(dict_args)

        # initialize weights
        # following https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf
        for name, param in self.rnn.named_parameters():
            if name.find("weight") > -1:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, seq_word_indexes, seq_lengths,
                seq_enc_states, seq_enc_hidden):
        """Forward pass."""
        # seqlen x batch size x embedding dim
        seq_word_embds = self.target_word_embd(seq_word_indexes)

        # init output tensor
        seqlen, batch_size, _ = seq_word_embds.size()
        log_probs = torch.zeros([seqlen, batch_size, self.vocab_size])
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        # init decoder hidden state
        if self.model_type == 'gru':
            self.hidden = self.init_hidden(seq_enc_hidden).view(
                self.num_layers, batch_size, self.hidden_size)
        elif self.model_type == 'lstm':
            hidden1 = self.init_hidden1(seq_enc_hidden).view(
                self.num_layers, batch_size, self.hidden_size)
            hidden2 = self.init_hidden2(seq_enc_hidden).view(
                self.num_layers, batch_size, self.hidden_size)
            self.hidden = (hidden1, hidden2)

        if self.attention:
            # batch_size x enc_hidden_dim x seqlen
            seq_enc_states = seq_enc_states.permute(1, 2, 0)

            if self.model_type == 'gru':
                hidden = self.hidden
            elif self.model_type == 'lstm':
                hidden = self.hidden[0]

            context, _ = self.attn_layer(
                seqlen, hidden.view(1, batch_size, -1),
                seq_enc_states)
        else:
            # seqlen
            context = seq_enc_states.expand(seqlen, -1, -1)


        # prep input
        seq_lengths, orig2sorted = seq_lengths.sort(0, descending=True)
        _, sorted2orig = orig2sorted.sort(0, descending=False)
        context_input = torch.cat([seq_word_embds, context], dim=2)
        context_input = context_input[:, orig2sorted, :]
        context_input = pack_padded_sequence(context_input, seq_lengths)

        # forward rnn and decode output sort
        output, _ = self.rnn(context_input, self.hidden)
        output = pad_packed_sequence(output, total_length=seqlen)
        output = output[0][:, sorted2orig, :]

        for i in range(seqlen):
            log_probs[i] = F.log_softmax(self.hidden2vocab(output[i]), dim=1)

        return log_probs.permute(1, 2, 0)  # batch size x vocab size x seqlen
