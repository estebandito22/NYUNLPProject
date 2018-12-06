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

        self.hiddenproj = nn.Linear(
            self.hidden_size, self.context_dim, bias=False)

    def forward(self, seqlen, hidden, contextvects, padding_mask=None):
        """Forward pass."""
        # contextvects: context_size x batch_size x context_dim
        # hidden: batch_size x hidden_size
        # padding mask: context_size x batch_size
        # init context
        context = torch.zeros(
            [seqlen, hidden.size(0), self.context_dim])
        attentions = torch.zeros(
            [seqlen, hidden.size(0), self.context_size])
        if torch.cuda.is_available():
            context = context.cuda()
            attentions = attentions.cuda()
        # create directed graph for attention at each time step
        for i in range(seqlen):
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
            attentions[i] = attn_weights.t().detach()
            # weighted sum of contextvects
            # outputs: batch_size x context_dim
            context[i] = (attn_weights.unsqueeze(2) * contextvects).sum(dim=0)

        return context, attentions
