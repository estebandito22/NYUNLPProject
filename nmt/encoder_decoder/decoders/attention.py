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

        self.attn = nn.Linear(self.context_dim, self.hidden_size)

    def forward(self, seqlen, hidden, contextvects):
        """Forward pass."""
        # contextvects: batch_size x context_dim x context_size
        # hidden: 1 x batch_size x hidden_size
        # init context
        context = torch.zeros(
            [seqlen, hidden.size(1), self.context_dim])
        if torch.cuda.is_available():
            context = context.cuda()
        # create directed graph for attention at each time step
        for i in range(seqlen):
            # batch_size x context_size x hidden_size
            context_proj = self.attn(contextvects.permute(0, 2, 1))
            # multiplies by batch,
            # (context_size x hidden_size) mv (hidden_size x 1)
            # -> batch_size x context_size x 1
            attn_weights = F.softmax(torch.matmul(
                context_proj, hidden.permute(1, 2, 0)), dim=1)
            # mutiplies by batch,
            # (context_dim x context_size) mv (context_size x 1)
            # -> batch_size x context_dim
            context[i] = torch.matmul(
                contextvects, attn_weights).squeeze()

        return context
