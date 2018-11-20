"""PyTorch class for a recurrent network sentence decoder."""

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.attention import AttentionMechanism
from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings


class GreedyDecoder(nn.Module):

    """Recurrent Decoder to decode encoded sentence."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentDecoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(GreedyDecoder, self).__init__()
        self.enc_hidden_dim = dict_args["enc_hidden_dim"]
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.hidden_size = dict_args["hidden_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.bos_idx = dict_args["bos_idx"]
        self.eos_idx = dict_args["eos_idx"]

        # lstm
        self.hidden = None

        # attention
        if self.attention:
            self.rnn = nn.GRUCell(
                self.enc_hidden_dim * 2 + self.word_embdim, self.hidden_size)

            dict_args = {'context_size': self.max_sent_len,
                         'context_dim': self.enc_hidden_dim * 2,
                         'hidden_size': self.hidden_size}
            self.attn_layer = AttentionMechanism(dict_args)

            self.init_hidden = nn.Linear(
                self.enc_hidden_dim * 2, self.hidden_size)
        else:
            self.rnn = nn.GRUCell(
                self.enc_hidden_dim + self.word_embdim, self.hidden_size)

            self.init_hidden = nn.Linear(self.enc_hidden_dim, self.hidden_size)

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

        # target embeddings
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size}
        self.target_word_embd = WordEmbeddings(dict_args)

    def _load_state(self, recurrent_decoder_state):
        # load pre-trained decoder state
        state_dict = OrderedDict()

        for k in recurrent_decoder_state.keys():
            if k.find('rnn.') > -1:
                key = k[:-3]
            else:
                key = k

            state_dict[key] = recurrent_decoder_state[k]

        self.load_state_dict(state_dict)

    def forward(self, seq_enc_states, seq_enc_hidden, recurrent_decoder_state):
        """Forward pass."""
        self._load_state(recurrent_decoder_state)
        self.hidden = self.init_hidden(seq_enc_hidden.view(1, -1))

        if self.attention:
            # 1 x enc_hidden_dim x seqlen
            seq_enc_states = seq_enc_states.permute(1, 2, 0)

        # init output tensor
        out_seq_indexes = []

        # initialize bos for forward decoding
        start_idx = torch.tensor([self.bos_idx]).unsqueeze(0)
        if torch.cuda.is_available():
            start_idx = start_idx.cuda()

        i_t = self.target_word_embd(start_idx)
        eos = False
        i = 0
        while eos is False and i < self.max_sent_len * 2:

            if self.attention:
                context = self.attn_layer(
                    1, self.hidden.unsqueeze(0), seq_enc_states)
            else:
                context = seq_enc_states.view(1, 1, -1)

            context_input = torch.cat([i_t, context], dim=2)[0]
            self.hidden = self.rnn(context_input, self.hidden)
            log_probs = F.log_softmax(self.hidden2vocab(self.hidden), dim=1)
            seq_index = log_probs.argmax(dim=1)

            if seq_index == self.eos_idx:
                eos = True
            else:
                out_seq_indexes.append(seq_index)
                i_t = self.target_word_embd(seq_index.unsqueeze(0))

            i += 1

        return out_seq_indexes  # (list) variable_seqlen
