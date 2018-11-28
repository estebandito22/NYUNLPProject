"""PyTorch class for a recurrent network sentence decoder."""

import torch
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.recurrent import RecurrentDecoder


class GreedyDecoder(RecurrentDecoder):

    """Recurrent Decoder to decode encoded sentence."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentDecoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        self.enc_hidden_dim = dict_args["enc_hidden_dim"]
        self.enc_num_layers = dict_args["enc_num_layers"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.num_layers = dict_args["num_layers"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.hidden_size = dict_args["hidden_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.bos_idx = dict_args["bos_idx"]
        self.eos_idx = dict_args["eos_idx"]
        self.model_type = dict_args["model_type"]

        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'enc_num_layers': self.enc_num_layers,
                     'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'num_layers': self.num_layers,
                     'dropout': self.dropout,
                     'vocab_size': self.vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.hidden_size,
                     'batch_size': self.batch_size,
                     'attention': self.attention,
                     'model_type': self.model_type}
        RecurrentDecoder.__init__(self, dict_args)

    def forward(self, seq_enc_states, seq_enc_hidden, recurrent_decoder_state):
        """Forward pass."""
        self.load_state_dict(recurrent_decoder_state)

        if self.model_type == 'gru':
            self.hidden = self.init_hidden(
                seq_enc_hidden.view(1, 1, -1)).view(
                    self.num_layers, 1, self.hidden_size)
        if self.model_type == 'lstm':
            hidden1 = self.init_hidden1(
                seq_enc_hidden.view(1, 1, -1)).view(
                    self.num_layers, 1, self.hidden_size)
            hidden2 = self.init_hidden2(
                seq_enc_hidden.view(1, 1, -1)).view(
                    self.num_layers, 1, self.hidden_size)
            self.hidden = (hidden1, hidden2)

        if self.attention:
            # 1 x enc_hidden_dim x seqlen
            seq_enc_states = seq_enc_states.permute(1, 2, 0)

        # init output tensor
        out_seq_indexes = []

        # initialize bos for forward decoding
        start_idx = torch.tensor([self.bos_idx]).view(1, -1)
        if torch.cuda.is_available():
            start_idx = start_idx.cuda()

        i_t = self.target_word_embd(start_idx)
        eos = False
        i = 0
        while eos is False and i < self.max_sent_len * 2:

            if self.attention:
                if self.model_type == 'gru':
                    hidden = self.hidden
                elif self.model_type == 'lstm':
                    hidden = self.hidden[0]

                context = self.attn_layer(
                    1, hidden.view(1, 1, -1), seq_enc_states)
            else:
                context = seq_enc_states.view(1, 1, -1)

            context_input = torch.cat([i_t, context], dim=2)
            output, self.hidden = self.rnn(context_input, self.hidden)
            log_probs = F.log_softmax(self.hidden2vocab(output[0]), dim=1)
            print(log_probs.argmax(dim=1))
            print(log_probs[0][log_probs.argmax(dim=1)])
            print(log_probs[1][log_probs.argmax(dim=1)])
            raise
            seq_index = log_probs.argmax(dim=1)

            if seq_index == self.eos_idx:
                eos = True
            else:
                out_seq_indexes.append(seq_index)
                i_t = self.target_word_embd(seq_index.unsqueeze(0))

            i += 1

        return out_seq_indexes  # (list) variable_seqlen
