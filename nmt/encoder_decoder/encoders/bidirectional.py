"""PyTorch classes for a recurrent network encoder."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.batch_size = dict_args["batch_size"]

        # GRU
        self.hidden = None
        self.init_hidden(self.batch_size)

        self.rnn = nn.GRU(
            input_size=self.word_embdim, hidden_size=self.hidden_size,
            num_layers=1, dropout=0, bidirectional=True)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        if torch.cuda.is_available():
            self.hidden = torch.zeros(2, batch_size, self.hidden_size).cuda()
        else:
            self.hidden = torch.zeros(2, batch_size, self.hidden_size)

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        _, hidden_batch_size, _ = self.hidden.size()
        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            detached_hidden = self.hidden.detach()
            detached_hidden.zero_()
            self.hidden = detached_hidden

    def forward(self, seq_word_embds, seq_lengths):
        """Forward pass."""
        seqlen, batch_size, _ = seq_word_embds.size()
        seq_lengths, orig2sorted = seq_lengths.sort(0, descending=True)
        _, sorted2orig = orig2sorted.sort(0, descending=False)
        seq_word_embds = seq_word_embds[:, orig2sorted, :]
        seq_word_embds = pack_padded_sequence(seq_word_embds, seq_lengths)

        out, h_n = self.rnn(seq_word_embds, self.hidden)

        out = pad_packed_sequence(out, total_length=seqlen)
        out = out[0][:, sorted2orig, :]

        h_n = h_n[:, sorted2orig, :]
        h_n = h_n.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return out, h_n
