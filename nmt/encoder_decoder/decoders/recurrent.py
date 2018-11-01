"""PyTorch class for a recurrent network sentence decoder."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nmt.encoder_decoder.decoders.attention import AttentionMechanism


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
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]

        # lstm
        self.hidden = None

        # attention
        if self.attention:
            self.rnn = nn.GRU(
                self.enc_hidden_dim * 2 + self.word_embdim, self.hidden_size,
                dropout=self.dropout)

            dict_args = {'context_size': self.max_sent_len,
                         'context_dim': self.enc_hidden_dim * 2,
                         'hidden_size': self.hidden_size}
            self.attn_layer = AttentionMechanism(dict_args)

            self.init_hidden = nn.Linear(
                self.enc_hidden_dim * 2, self.hidden_size)
        else:
            self.rnn = nn.GRU(
                self.enc_hidden_dim + self.word_embdim, self.hidden_size,
                dropout=self.dropout)

            self.init_hidden = nn.Linear(self.enc_hidden_dim, self.hidden_size)

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, seq_word_embds, seq_lengths,
                seq_enc_states, seq_enc_hidden):
        """Forward pass."""
        self.hidden = self.init_hidden(seq_enc_hidden).unsqueeze(0)
        # init output tensor
        seqlen, batch_size, _ = seq_word_embds.size()
        log_probs = torch.zeros([seqlen, batch_size, self.vocab_size])
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        if self.attention:
            # batch_size x enc_hidden_dim x seqlen
            seq_enc_states = seq_enc_states.permute(1, 2, 0)
            # batch_size x enc_hidden_dim
            context = self.attn_layer(seqlen, self.hidden, seq_enc_states)
        else:
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
