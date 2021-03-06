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

        self.attn = nn.Linear(self.hidden_size, self.context_dim)

    def forward(self, seqlen, hidden, contextvects, padding_mask=None):
        """Forward pass."""
        # contextvects: batch_size x context_dim x context_size
        # hidden: 1 x batch_size x hidden_size
        # padding mask: batch_size x maxseqlen
        # init context
        context = torch.zeros(
            [seqlen, hidden.size(1), self.context_dim])
        attentions = torch.zeros(
            [seqlen, hidden.size(1), self.context_size])
        if torch.cuda.is_available():
            context = context.cuda()
            attentions = attentions.cuda()
        # create directed graph for attention at each time step
        for i in range(seqlen):
            # batch_size x context_size x hidden_size
            context_proj = self.attn(contextvects.permute(0, 2, 1))
            # multiplies by batch,
            # (context_size x hidden_size) mv (hidden_size x 1)
            # -> batch_size x context_size x 1
            attn_scores = torch.matmul(context_proj, hidden.permute(1, 2, 0))
            # dont attend over padding
            if padding_mask is not None:
                attn_scores.masked_fill_(padding_mask, float('-inf'))
            # normalize attn scores
            attn_weights = F.softmax(attn_scores, dim=1)
            attentions[i] = attn_weights.squeeze()
            # mutiplies by batch,
            # (context_dim x context_size) mv (context_size x 1)
            # -> batch_size x context_dim
            context[i] = torch.matmul(
                contextvects, attn_weights).squeeze()

        return context, attentions
