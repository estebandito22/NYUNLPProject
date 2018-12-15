"""
PyTorch class to perform attention over context vectors.
Adapted from https://github.com/pytorch/fairseq
"""

import numpy as np
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
        self.kernel_size = dict_args['kernel_size']

        self.hiddenproj = nn.Linear(
            self.hidden_size, self.context_dim, bias=False)
        self.outputproj = nn.Linear(
            self.context_dim + self.hidden_size, self.context_dim, bias=False)

    def forward(self, hidden, contextvects, padding_mask=None):
        """Forward pass."""
        # contextvects: context_size x batch_size x context_dim
        # hidden: batch_size x hidden_size
        # padding mask: context_size x batch_size

        # to handle conv encoder attention mechanism
        if self.kernel_size != 0:
            contextvects, contextvects_apply = contextvects
            apply_vects = True
        else:
            apply_vects = False

        # oupututs: batch_size x context_dim
        hidden_proj = self.hiddenproj(hidden.squeeze(0))
        # outputs: context_size x batch_size
        attn_scores = (contextvects * hidden_proj.unsqueeze(0)).sum(dim=2)

        # dont attend over padding
        if padding_mask is not None:
            attn_scores.float().masked_fill_(
                padding_mask, float('-inf')).type_as(attn_scores)

        # normalize attn scores
        # outputs: contet_size x batch_size
        attn_weights = F.softmax(attn_scores, dim=0)

        # to handle conv encoder attention mechanism
        if apply_vects:
            contextvects = contextvects_apply

        # weighted sum of contextvects
        # outputs: batch_size x context_dim
        c = (attn_weights.unsqueeze(2) * contextvects).sum(dim=0)
        context = torch.tanh(self.outputproj(torch.cat((c, hidden), dim=1)))

        if torch.cuda.is_available():
            context = context.cuda()
            attn_weights = attn_weights.cuda()

        return context, attn_weights
