"""PyTorch class for a recurrent network sentence decoder."""

import random

import torch
from torch import nn
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.fairseq import FairseqDecoder
from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class FairseqGreedyDecoder(FairseqDecoder):

    """Recurrent Decoder to decode encoded sentence."""

    def __init__(self, dict_args):
        """
        Initialize FairseqGreedyDecoder.

        Args
            dict_args: dictionary containing the following keys:
        """
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
        FairseqDecoder.__init__(self, dict_args)

    def forward(self, seq_enc_states, enc_padding_mask, seq_enc_hidden,
                recurrent_decoder_state):
        """Forward pass."""
        self.load_state_dict(recurrent_decoder_state)
        batch_size = seq_enc_states.size(1)

        # init decoder hidden state
        if self.kernel_size == 0:
            if self.model_type == 'gru':
                prev_hiddens = [seq_enc_hidden[i] for i in range(self.num_layers)]
                prev_cells = None
            elif self.model_type == 'lstm':
                prev_hiddens = [seq_enc_hidden[0][i] for i in range(self.num_layers)]
                prev_cells = [seq_enc_hidden[1][i] for i in range(self.num_layers)]
        else:
            prev_hiddens, prev_cells = self.init_hidden(seq_enc_hidden)

        # init context
        context = seq_enc_states.data.new(batch_size, self.enc_hidden_dim * self.enc_num_directions)

        # init attention scores
        srclen = seq_enc_states.size(0)
        attn_scores = seq_enc_states.data.new(srclen, self.max_sent_len * 2, batch_size).zero_()

        # init output tensor
        out_seq_indexes = []

        # initialize bos for forward decoding
        # batch_size x 1
        start_idx = torch.LongTensor(size=[batch_size, 1]).fill_(self.bos_idx)
        if torch.cuda.is_available():
            start_idx = start_idx.cuda()
        i_t = self.target_word_embd(start_idx).squeeze(0)
        i_t = self.drop_in(i_t)

        eos = False
        j = 0
        while eos is False and j < self.max_sent_len * 2:

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
                context = out
            else:
                out = hidden
                context = self.context_proj(seq_enc_states.permute(1, 0, 2).contiguous().view(batch_size, -1))
            out = self.drop_out(out)

            # input for next time step
            log_probs = F.log_softmax(self.hidden2vocab(out), dim=1)
            seq_index = log_probs.argmax(dim=1).detach() # detach from history as input

            if seq_index == self.eos_idx:
                eos = True
            else:
                out_seq_indexes.append(seq_index)
                i_t = self.target_word_embd(seq_index.unsqueeze(1)).squeeze(0)
                i_t = self.drop_in(i_t)

            j += 1

        return out_seq_indexes, attn_scores  # batch size x vocab size x seqlen
