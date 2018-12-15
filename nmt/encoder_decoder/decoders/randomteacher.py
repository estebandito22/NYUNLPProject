"""
PyTorch class for a recurrent network sentence decoder.
Inspired by https://github.com/pytorch/fairseq
"""

import random

import torch
from torch import nn
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.bhattention import AttentionMechanism
from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class RandomTeacherDecoder(nn.Module):

    """Recurrent Decoder to decode encoded sentence."""

    def __init__(self, dict_args):
        """
        Initialize RandomTeacherDecoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(RandomTeacherDecoder, self).__init__()
        self.enc_hidden_dim = dict_args["enc_hidden_dim"]
        self.enc_num_layers = dict_args["enc_num_layers"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.dropout_in = dict_args["dropout_in"]
        self.dropout_out = dict_args["dropout_out"]
        self.vocab_size = dict_args["vocab_size"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.hidden_size = dict_args["hidden_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.bos_idx = dict_args["bos_idx"]
        self.eos_idx = dict_args["eos_idx"]
        self.model_type = dict_args["model_type"]
        self.tf_ratio = dict_args["tf_ratio"]
        self.kernel_size = dict_args["kernel_size"]
        self.enc_num_directions = 2 if self.attention and self.kernel_size == 0 else 1

        # init recurrent layers
        if self.model_type == 'gru':
            self.layers = nn.ModuleList([
                nn.GRUCell(
                    input_size=self.enc_hidden_dim
                    * self.enc_num_directions
                    + self.word_embdim if layer == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                )
                for layer in range(self.num_layers)
            ])

        elif self.model_type == 'lstm':
            self.layers = nn.ModuleList([
                nn.LSTMCell(
                    input_size=self.enc_hidden_dim
                    * self.enc_num_directions
                    + self.word_embdim if layer == 0 else self.hidden_size,
                    hidden_size=self.hidden_size,
                )
                for layer in range(self.num_layers)
            ])

        # attention
        if self.attention:
            nd = self.enc_num_directions
            dict_args = {'context_size': self.max_sent_len,
                         'context_dim': self.enc_hidden_dim * nd,
                         'hidden_size': self.hidden_size,
                         'kernel_size': self.kernel_size}
            self.attn_layer = AttentionMechanism(dict_args)
        else:
            self.context_proj = nn.Linear(
                self.enc_hidden_dim * self.enc_num_layers, self.enc_hidden_dim)

        # mlp output
        if self.attention:
            self.hidden2vocab = nn.Linear(
                self.enc_hidden_dim * self.enc_num_directions, self.vocab_size)
        else:
            self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

        # target embeddings
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size,
                     'kernel_size': self.kernel_size}
        self.target_word_embd = WordEmbeddings(dict_args)

        # dropout layers
        self.drop = nn.Dropout(p=self.dropout)
        self.drop_in = nn.Dropout(p=self.dropout_in)
        self.drop_out = nn.Dropout(p=self.dropout_out)

        # # register hooks for reporting gradients
        # self.ih_hooks = [x.weight_ih.register_hook(lambda grad: print("ih_grad", grad)) for x in self.layers]
        # self.hh_hooks = [x.weight_hh.register_hook(lambda grad: print("hh_grad", grad)) for x in self.layers]

        # initialize hiddens for conv model
        if self.kernel_size != 0:
            for layer in self.layers:
                nn.init.uniform_(layer.weight_ih, -0.05, 0.05)
                nn.init.uniform_(layer.weight_hh, -0.05, 0.05)
                nn.init.constant_(layer.bias_ih, 0)
                nn.init.constant_(layer.bias_hh, 0)

    def init_hidden(self, seq_enc_hidden):
        """Initialize the hidden state of the RNN."""

        if self.kernel_size == 0:
            # hidden state initialization for recurren encoders
            if self.model_type == 'gru':
                prev_hiddens = [seq_enc_hidden[i]
                                for i in range(self.num_layers)]
                prev_cells = None
            elif self.model_type == 'lstm':
                prev_hiddens = [seq_enc_hidden[0][i]
                                for i in range(self.num_layers)]
                prev_cells = [seq_enc_hidden[1][i]
                              for i in range(self.num_layers)]

        else:
            # hidden state initialization for convolutional encoder
            if self.model_type == 'gru':
                prev_hiddens = [torch.zeros_like(seq_enc_hidden)
                                for _ in range(self.num_layers)]
                prev_cells = None
            elif self.model_type == 'lstm':
                prev_hiddens = [torch.zeros_like(seq_enc_hidden[0])
                                for _ in range(self.num_layers)]
                prev_cells = [torch.zeros_like(seq_enc_hidden[1])
                              for _ in range(self.num_layers)]

        return prev_hiddens, prev_cells

    def forward(self, seq_word_indexes, seq_lengths,
                seq_enc_states, enc_padding_mask, seq_enc_hidden):
        """Forward pass."""
        # get target word embeddings
        seq_word_embds = self.target_word_embd(seq_word_indexes)

        # init output tensor
        seqlen, batch_size, _ = seq_word_embds.size()
        log_probs = []

        # init decoder hidden state
        prev_hiddens, prev_cells = self.init_hidden(seq_enc_hidden)

        # init context
        context = seq_word_embds.data.new(
            batch_size, self.enc_hidden_dim * self.enc_num_directions).zero_()

        # init attention scores
        if self.kernel_size != 0:
            srclen = seq_enc_states[0].size(0)
        else:
            srclen = seq_enc_states.size(0)
        attn_scores = seq_word_embds.data.new(
            srclen, seqlen, batch_size).zero_()

        use_tf = True if random.random() < self.tf_ratio else False
        if not use_tf:
            # initialize bos for forward decoding
            # batch_size x 1
            start_idx = torch.LongTensor(
                size=[batch_size, 1]).fill_(self.bos_idx)
            if torch.cuda.is_available():
                start_idx = start_idx.cuda()
            i_t = self.target_word_embd(start_idx).squeeze(0)
            i_t = self.drop_in(i_t)

            for j in range(seqlen):
                input = torch.cat([i_t, context], dim=1)

                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    if self.model_type == 'gru':
                        hidden = rnn(input, prev_hiddens[i])
                    elif self.model_type == 'lstm':
                        hidden, cell = rnn(
                            input, (prev_hiddens[i], prev_cells[i]))
                    # hidden state becomes the input to the next layer
                    input = self.drop(hidden)
                    # save state for next time step
                    if self.model_type == 'gru':
                        prev_hiddens[i] = hidden
                    elif self.model_type == 'lstm':
                        prev_hiddens[i] = hidden
                        prev_cells[i] = cell

                # apply attention using the last layer's hidden state
                if self.attention:
                    out, attn_scores[:, j, :] = self.attn_layer(
                        hidden, seq_enc_states, enc_padding_mask)
                    context = out
                else:
                    out = hidden
                    if self.kernel_size == 0:
                        context = self.context_proj(
                            seq_enc_states.permute(1, 0, 2).contiguous().
                            view(batch_size, -1))
                    else:
                        context = seq_enc_states
                out = self.drop_out(out)

                # input for next time step
                log_prob = F.log_softmax(self.hidden2vocab(out), dim=1)
                log_probs += [log_prob]
                seq_index = log_prob.argmax(dim=1).detach()
                # detach from history as input
                i_t = self.target_word_embd(seq_index.unsqueeze(1)).squeeze(0)
                i_t = self.drop_in(i_t)

        else:
            for j in range(seqlen):

                i_t = seq_word_embds[j]
                i_t = self.drop_in(i_t)
                input = torch.cat([i_t, context], dim=1)

                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    if self.model_type == 'gru':
                        hidden = rnn(input, prev_hiddens[i])
                    elif self.model_type == 'lstm':
                        hidden, cell = rnn(
                            input, (prev_hiddens[i], prev_cells[i]))
                    # hidden state becomes the input to the next layer
                    input = self.drop(hidden)
                    # save state for next time step
                    if self.model_type == 'gru':
                        prev_hiddens[i] = hidden
                    elif self.model_type == 'lstm':
                        prev_hiddens[i] = hidden
                        prev_cells[i] = cell

                # apply attention using the last layer's hidden state
                if self.attention:
                    out, attn_scores[:, j, :] = self.attn_layer(
                        hidden, seq_enc_states, enc_padding_mask)
                    context = out
                else:
                    out = hidden
                    if self.kernel_size == 0:
                        context = self.context_proj(
                            seq_enc_states.permute(1, 0, 2).contiguous()
                            .view(batch_size, -1))
                    else:
                        context = seq_enc_states
                out = self.drop_out(out)

                # input for next time step
                log_prob = F.log_softmax(self.hidden2vocab(out), dim=1)
                seq_index = log_prob.argmax(dim=1).detach()
                # detach from history as input
                log_probs += [log_prob]

        log_probs = torch.stack(log_probs)
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        return log_probs.permute(1, 2, 0)  # batch size x vocab size x seqlen
