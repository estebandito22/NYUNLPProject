"""PyTorch class for a recurrent network sentence decoder."""

import random

import torch
from torch import nn
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.bhattention import AttentionMechanism
from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class FairseqDecoder(nn.Module):

    """Recurrent Decoder to decode encoded sentence."""

    def __init__(self, dict_args):
        """
        Initialize FairseqDecoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(FairseqDecoder, self).__init__()
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

        # attention
        if self.attention:
            if self.model_type == 'gru':
                self.layers = nn.ModuleList([
                    nn.GRUCell(
                        input_size=self.enc_hidden_dim * 2 + self.word_embdim if layer == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                    )
                    for layer in range(self.num_layers)
                ])

            elif self.model_type == 'lstm':
                self.layers = nn.ModuleList([
                    nn.LSTMCell(
                        input_size=self.enc_hidden_dim * 2 + self.word_embdim if layer == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                    )
                    for layer in range(self.num_layers)
                ])

            dict_args = {'context_size': self.max_sent_len,
                         'context_dim': self.enc_hidden_dim * 2,
                         'hidden_size': self.hidden_size}
            self.attn_layer = AttentionMechanism(dict_args)

        else:
            if self.model_type == 'gru':
                self.layers = nn.ModuleList([
                    nn.GRUCell(
                        input_size=self.enc_num_layers * self.enc_hidden_dim + self.word_embdim if layer == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                    )
                    for layer in range(self.num_layers)
                ])

            elif self.model_type == 'lstm':
                self.layers = nn.ModuleList([
                    nn.LSTMCell(
                        input_size=self.enc_num_layers * self.enc_hidden_dim + self.word_embdim if layer == 0 else self.hidden_size,
                        hidden_size=self.hidden_size,
                    )
                    for layer in range(self.num_layers)
                ])

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

        # target embeddings
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'vocab_size': self.vocab_size}
        self.target_word_embd = WordEmbeddings(dict_args)

        self.drop = nn.Dropout(p=self.dropout)
        self.drop_in = nn.Dropout(p=self.dropout_in)
        self.drop_out = nn.Dropout(p=self.dropout_out)

    def forward(self, seq_word_indexes, seq_lengths,
                seq_enc_states, enc_padding_mask, seq_enc_hidden):
        """Forward pass."""
        # get target word embeddings
        seq_word_embds = self.target_word_embd(seq_word_indexes)

        # init output tensor
        seqlen, batch_size, _ = seq_word_embds.size()
        # log_probs = torch.zeros([seqlen, batch_size, self.vocab_size])
        # if torch.cuda.is_available():
        #     log_probs = log_probs.cuda()
        log_probs = []

        # init decoder hidden state
        if self.model_type == 'gru':
            prev_hiddens = [seq_enc_hidden[i] for i in range(self.num_layers)]
        elif self.model_type == 'lstm':
            prev_hiddens = [seq_enc_hidden[0][i] for i in range(self.num_layers)]
            prev_cells = [seq_enc_hidden[1][i] for i in range(self.num_layers)]
        context = seq_word_embds.data.new(batch_size, self.enc_hidden_dim * 2)

        # init attention scores
        srclen = seq_enc_states.size(0)
        attn_scores = seq_word_embds.data.new(srclen, seqlen, batch_size).zero_()

        use_tf = True if random.random() < self.tf_ratio else False
        if not use_tf:
            # initialize bos for forward decoding
            # batch_size x 1
            start_idx = torch.LongTensor(size=[batch_size, 1]).fill_(self.bos_idx)
            if torch.cuda.is_available():
                start_idx = start_idx.cuda()
            i_t = self.target_word_embd(start_idx).squeeze()
            i_t = self.drop_in(i_t)

            for j in range(seqlen):

                input = torch.cat([i_t, context], dim=1)

                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    if self.model_type == 'gru':
                        hidden = rnn(input, prev_hiddens[i])
                    elif self.model_type == 'lstm':
                        hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
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
                    out, attn_scores[:, j, :] = self.attn_layer(hidden, seq_enc_states, enc_padding_mask)
                else:
                    out = hidden
                out = self.drop_out(out)

                # context for next time step
                context = out

                # input for next time step
                log_prob = F.log_softmax(self.hidden2vocab(out), dim=1)
                log_probs += [log_prob]
                seq_index = log_prob.argmax(dim=1).detach() # detach from history as input
                i_t = self.target_word_embd(seq_index.unsqueeze(1)).squeeze()
                i_t = self.drop_in(i_t)

        else:
            for j in range(seqlen):

                i_t = seq_word_embds[j].squeeze()
                i_t = self.drop_in(i_t)
                input = torch.cat([i_t, context], dim=1)

                for i, rnn in enumerate(self.layers):
                    # recurrent cell
                    if self.model_type == 'gru':
                        hidden = rnn(input, prev_hiddens[i])
                    elif self.model_type == 'lstm':
                        hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
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
                    out, attn_scores[:, j, :] = self.attn_layer(hidden, seq_enc_states, enc_padding_mask)
                else:
                    out = hidden
                out = self.drop_out(out)

                # context for next time step
                context = out

                # input for next time step
                log_prob = F.log_softmax(self.hidden2vocab(out), dim=1)
                seq_index = log_prob.argmax(dim=1).detach()  # detach from history as input
                log_probs += [log_prob]

        log_probs = torch.stack(log_probs)
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        return log_probs.permute(1, 2, 0)  # batch size x vocab size x seqlen
