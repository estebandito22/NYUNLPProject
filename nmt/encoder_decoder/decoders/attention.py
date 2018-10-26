"""PyTorch class to perform attention over context vectors."""

import torch
from torch import nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):

    """Implements an attention mechanism."""

    def __init__(self, dict_args):
        """
        Initialize AttentionMechanism.

        Args
            dict_args: dictionary containing the following keys:
                context_size: the dimension of the context vectors to perform
                             attention over.
                hidden_size: the size of the hidden state.
                input_dim: the dimension the input.
        """
        super(AttentionMechanism, self).__init__()
        self.context_size = dict_args['context_size']
        self.context_dim = dict_args['context_dim']
        self.hidden_size = dict_args['hidden_size']
        self.word_embdim = dict_args['word_embdim']

        self.attn = nn.Linear(
            self.hidden_size + self.word_embdim, self.context_size)

    def forward(self, seq_word_embds, hidden, contextvects):
        """Forward pass."""
        # contextvects: batch_size x context_dim x context_size
        # seq_word_embds: seqlen x batch_size x word_embdim
        seqlen, batch_size, _ = seq_word_embds.size()
        # seqlen x batch_size x word_embdim + hidden_size
        attn_input = torch.cat(
            [seq_word_embds, hidden.expand(seqlen, -1, -1)], dim=2)
        # init context
        context = torch.zeros(
            [seqlen, batch_size, self.context_dim])
        if torch.cuda.is_available():
            context = context.cuda()
        # create directed graph for attention at each time step
        for i in range(seqlen):
            # batch_size x word_embdim + hidden_size
            # -> batch_size x context_size
            attn_weights = F.softmax(self.attn(attn_input[i]), dim=1)
            # mutiplies by batch,
            # (context_dim x context_size) mm (context_size x 1)
            # giving (context_dim, )
            context[i] = torch.matmul(
                contextvects, attn_weights.unsqueeze(2)).squeeze()

        return context
