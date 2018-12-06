"""PyTorch class for a recurrent network sentence decoder."""

import random

import torch
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.recurrent import RecurrentDecoder


class RandomTeacherDecoder(RecurrentDecoder):

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

        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'enc_num_layers': self.enc_num_layers,
                     'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings,
                     'num_layers': self.num_layers,
                     'dropout': self.dropout,
                     'dropout_in': self.dropout_in,
                     'dropout_out': self.dropout_out,
                     'vocab_size': self.vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.hidden_size,
                     'batch_size': self.batch_size,
                     'attention': self.attention,
                     'model_type': self.model_type}
        RecurrentDecoder.__init__(self, dict_args)

    def forward(self, seq_word_indexes, seq_lengths,
                seq_enc_states, enc_padding_mask, seq_enc_hidden):
        """Forward pass."""
        # get target word embeddings
        seq_word_embds = self.target_word_embd(seq_word_indexes)

        # init output tensor
        seqlen, batch_size, _ = seq_word_embds.size()
        log_probs = torch.zeros([seqlen, batch_size, self.vocab_size])
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        # init decoder hidden state
        if self.model_type == 'gru':
            self.hidden = self.init_hidden(seq_enc_hidden).view(
                self.num_layers, batch_size, self.hidden_size)
        elif self.model_type == 'lstm':
            hidden1 = self.init_hidden1(seq_enc_hidden).view(
                self.num_layers, batch_size, self.hidden_size)
            hidden2 = self.init_hidden2(seq_enc_hidden).view(
                self.num_layers, batch_size, self.hidden_size)
            self.hidden = (hidden1, hidden2)

        # prep encoder states once for attention
        # if self.attention:
        #     # batch_size x enc_hidden_dim x seqlen
        #     seq_enc_states = seq_enc_states.permute(1, 2, 0)

        use_tf = True if random.random() < self.tf_ratio else False
        if not use_tf:
            # initialize bos for forward decoding
            # batch_size x 1
            start_idx = torch.LongTensor(size=[batch_size, 1]).fill_(self.bos_idx)
            if torch.cuda.is_available():
                start_idx = start_idx.cuda()
            i_t = self.target_word_embd(start_idx)
            i_t = self.drop_in(i_t)

            for i in range(seqlen):

                # perform attention at time step
                if self.attention:
                    if self.model_type == 'gru':
                        hidden = self.hidden
                    elif self.model_type == 'lstm':
                        hidden = self.hidden[0]

                    context, _ = self.attn_layer(
                        1, hidden.view(batch_size, -1), seq_enc_states,
                        enc_padding_mask)
                else:
                    context = seq_enc_states.unsqueeze(0)

                context_input = torch.cat([i_t, context], dim=2)
                output, self.hidden = self.rnn(context_input, self.hidden)
                output = self.drop_out(output)
                log_probs[i] = F.log_softmax(self.hidden2vocab(output[0]), dim=1)
                seq_index = log_probs[i].argmax(dim=1).detach() # detach from history as input

                i_t = self.target_word_embd(seq_index.unsqueeze(1))
                i_t = self.drop_in(i_t)

        else:
            for i in range(seqlen):
                i_t = seq_word_embds[i].unsqueeze(0)
                i_t = self.drop_in(i_t)

                # perform attention at time step
                if self.attention:
                    if self.model_type == 'gru':
                        hidden = self.hidden
                    elif self.model_type == 'lstm':
                        hidden = self.hidden[0]

                    context, _ = self.attn_layer(
                        1, hidden.view(batch_size, -1), seq_enc_states,
                        enc_padding_mask)
                else:
                    context = seq_enc_states.unsqueeze(0)

                context_input = torch.cat([i_t, context], dim=2)
                output, self.hidden = self.rnn(context_input, self.hidden)
                log_probs[i] = F.log_softmax(self.hidden2vocab(output[0]), dim=1)

        return log_probs.permute(1, 2, 0)  # batch size x vocab size x seqlen
