"""PyTorch class for a recurrent network sentence decoder."""

from queue import PriorityQueue

import torch
import torch.nn.functional as F

from nmt.encoder_decoder.decoders.recurrent import RecurrentDecoder


class Beam(object):

    """Keeps track of details related to a single beam"""

    def __init__(self, log_prob, sequence, input_t, attentions, hidden):
        self.log_prob = log_prob
        self.sequence = sequence
        self.input_t = input_t
        self.attentions = attentions
        self.hidden = hidden

    def __lt__(self, other):
        return self.log_prob < other.log_prob


class BeamDecoder(RecurrentDecoder):

    """Recurrent Decoder to decode encoded sentence with Beam search."""

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

    def forward(self, seq_enc_states, seq_enc_hidden, recurrent_decoder_state, B=1):
        """
        Forward pass with beam search.

        Args
            B: (int) beam width
        """
        self.load_state_dict(recurrent_decoder_state)

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
        i_t = self.drop_in(i_t)

        eos = False
        i = 0
        beam_nodes = PriorityQueue()
        while eos is False and i < self.max_sent_len * 2:

            # init beams if dont exists/get top beams
            if beam_nodes.empty():
                top_B_beams = []
                for _ in range(B):
                    if self.model_type == 'gru':
                        hidden = self.init_hidden(
                            seq_enc_hidden.view(1, 1, -1)).view(
                                self.num_layers, 1, self.hidden_size)
                    if self.model_type == 'lstm':
                        hidden1 = self.init_hidden1(
                            seq_enc_hidden.view(1, 1, -1)).view(
                                self.num_layers, 1, self.hidden_size)
                        hidden2 = self.init_hidden2(
                            seq_enc_hidden.view(1, 1, -1)).view(
                                self.num_layers, 1, self.hidden_size)
                        hidden = (hidden1, hidden2)

                    top_B_beams.append(Beam(float('inf'), [], i_t, [], hidden))
            else:
                top_B_beams = [beam_nodes.get()[1] for _ in range(B)]

            eos = True
            for top_beam in top_B_beams:
                # If all top beams end with <eos> then the search is done.
                last_idx = top_beam.sequence[-1] if top_beam.sequence else None
                if last_idx != self.eos_idx:
                    eos = False

                    # get beam hidden
                    hidden = top_beam.hidden

                    # create context
                    if self.attention:
                        if self.model_type == 'gru':
                            h_t = hidden
                        elif self.model_type == 'lstm':
                            (h_t, c_t) = hidden
                        context, attentions = self.attn_layer(
                            1, h_t.view(1, 1, -1), seq_enc_states)
                    else:
                        context = seq_enc_states.view(1, 1, -1)

                    # add attentions
                    beam_attentions = top_beam.attentions
                    beam_attentions += [attentions.squeeze()]

                    i_t = top_beam.input_t
                    i_t = self.drop_in(i_t)
                    context_input = torch.cat([i_t, context], dim=2)
                    output, hidden = self.rnn(context_input, hidden)
                    output = self.drop_out(output)
                    log_probs = F.log_softmax(self.hidden2vocab(output[0]), dim=1)

                    # Perform beam search
                    top_B = torch.topk(log_probs, B)
                    # print(top_B)
                    beam_log_probs_neg, beam_seq_indices = top_B
                    beam_log_probs = beam_log_probs_neg * -1  # for priority queue

                    for _b in range(B):
                        beam_log_prob = beam_log_probs.cpu().numpy()[0][_b]
                        beam_seq_index = beam_seq_indices[0][_b].unsqueeze(0)
                        new_seq = top_beam.sequence + [beam_seq_index]
                        new_log_prob = top_beam.log_prob + beam_log_prob
                        new_input_t = self.target_word_embd(beam_seq_index.unsqueeze(0))
                        possible_top_beam = Beam(new_log_prob, new_seq, new_input_t, beam_attentions, hidden)
                        beam_nodes.put( (new_log_prob, possible_top_beam) )
            # if i > 3:
            #     print(beam_nodes.get())
            #     raise

            i += 1
        best_beam = beam_nodes.get()[1]
        out_seq_indexes = best_beam.sequence
        out_seq_attentions = best_beam.attentions
        return out_seq_indexes, out_seq_attentions
