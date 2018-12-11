"""PyTorch class for a recurrent network sentence decoder."""

from queue import PriorityQueue

import torch
import torch.nn.functional as F

import numpy as np

from nmt.encoder_decoder.decoders.randomteacher import RandomTeacherDecoder


class Beam(object):

    """Keeps track of details related to a single beam"""

    def __init__(self, log_prob, sequence, input_t, context, prev_hiddens, prev_cells=None, attentions=None):
        self.log_prob = log_prob
        self.sequence = sequence
        self.input_t = input_t
        self.context = context
        self.attentions = attentions
        self.prev_hiddens = prev_hiddens
        self.prev_cells = prev_cells

    def __lt__(self, other):
        return self.log_prob < other.log_prob


class BeamDecoder(RandomTeacherDecoder):

    """Recurrent Decoder to decode encoded sentence with Beam search."""

    def __init__(self, dict_args):
        """
        Initialize BeamDecoder.

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
        self.kernel_size = dict_args["kernel_size"]
        RandomTeacherDecoder.__init__(self, dict_args)

    @staticmethod
    def _normalize_length(current_length, alpha=0.5):
        # http://opennmt.net/OpenNMT/translation/beam_search/#length-normalization
        num = (5 + np.abs(current_length))**alpha
        den = (5 + 1)**alpha
        return num / den

    def forward(self, seq_enc_states, enc_padding_mask, seq_enc_hidden,
                recurrent_decoder_state, B=1):
        """
        Forward pass with beam search.

        Args
            B: (int) beam width
        """
        self.load_state_dict(recurrent_decoder_state)

        # init output tensor
        out_seq_indexes = []

        # initialize bos for forward decoding
        # batch_size x 1
        start_idx = torch.LongTensor(size=[1, 1]).fill_(self.bos_idx)
        if torch.cuda.is_available():
            start_idx = start_idx.cuda()
        i_t = self.target_word_embd(start_idx).squeeze(0)
        i_t = self.drop_in(i_t)

        eos = False
        j = 0
        beam_nodes = PriorityQueue()
        while eos is False and j < self.max_sent_len * 2:

            # init beams if dont exists/get top beams
            if beam_nodes.empty():
                top_B_beams = []
                for _ in range(B):
                    # init decoder hidden state
                    prev_hiddens, prev_cells = self.init_hidden(seq_enc_hidden)

                    # init context and attention scores
                    if self.kernel_size != 0:
                        context = seq_enc_states[0].data.new(
                            1, self.enc_hidden_dim * self.enc_num_directions).\
                            zero_()
                        srclen = seq_enc_states[0].size(0)
                        attn_scores = seq_enc_states[0].data.new(
                            srclen, self.max_sent_len * 2, 1).zero_()
                    else:
                        context = seq_enc_states.data.new(
                            1, self.enc_hidden_dim * self.enc_num_directions).\
                            zero_()
                        srclen = seq_enc_states.size(0)
                        attn_scores = seq_enc_states.data.new(
                            srclen, self.max_sent_len * 2, 1).zero_()

                    top_B_beams.append(Beam(float('inf'), [], i_t, context,
                                            prev_hiddens, prev_cells,
                                            attn_scores))
            else:
                top_B_beams = [beam_nodes.get()[1] for _ in range(B)]
                with beam_nodes.mutex:
                    beam_nodes.queue.clear() # Clear the remainder of the queue

            eos = True
            for top_beam in top_B_beams:
                # If all top beams end with <eos> then the search is done.
                last_idx = top_beam.sequence[-1] if top_beam.sequence else None
                if last_idx != self.eos_idx:
                    eos = False

                    # get beam hiddens, cells
                    prev_hiddens = top_beam.prev_hiddens
                    prev_cells = top_beam.prev_cells

                    # get beam input and context
                    i_t = top_beam.input_t
                    context = top_beam.context

                    # get beam attn_scores
                    attn_scores = top_beam.attentions

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
                                seq_enc_states.permute(1, 0, 2).contiguous().
                                view(1, -1))
                        else:
                            context = seq_enc_states
                    out = self.drop_out(out)

                    # Perform beam search
                    log_probs = F.log_softmax(self.hidden2vocab(out), dim=1)
                    top_B = torch.topk(log_probs, B, dim=1)
                    # print(top_B)
                    beam_log_probs_neg, beam_seq_indices = top_B
                    # for priority queue
                    beam_log_probs = beam_log_probs_neg * -1

                    for _b in range(B):
                        beam_log_prob = beam_log_probs.cpu().numpy()[0][_b]
                        beam_log_prob /= self._normalize_length(j + 1)
                        beam_seq_index = beam_seq_indices[0][_b].unsqueeze(0)
                        new_seq = top_beam.sequence + [beam_seq_index]
                        if j == 0:
                            # Don't take initialized beam probs
                            new_log_prob = beam_log_prob
                        else:
                            new_log_prob = top_beam.log_prob + beam_log_prob
                        new_context = context
                        new_prev_hiddens = prev_hiddens
                        new_prev_cells = prev_cells
                        new_attn_scores = attn_scores
                        new_input_t = self.target_word_embd(
                            beam_seq_index.unsqueeze(1)).squeeze(0)
                        possible_top_beam = Beam(
                            new_log_prob, new_seq, new_input_t, new_context,
                            new_prev_hiddens, new_prev_cells, new_attn_scores)
                        beam_nodes.put( (new_log_prob, possible_top_beam) )

            # increment the step in the sequence
            j += 1

        # return the best beam and its attentions
        best_beam = beam_nodes.get()[1]
        out_seq_indexes = best_beam.sequence
        out_seq_attentions = best_beam.attentions
        return out_seq_indexes, out_seq_attentions
