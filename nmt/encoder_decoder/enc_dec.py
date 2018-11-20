"""PyTorch classe for encoder-decoder netowork."""

from torch import nn

from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings

from nmt.encoder_decoder.encoders.recurrent import RecurrentEncoder
from nmt.encoder_decoder.encoders.bidirectional import BidirectionalEncoder

from nmt.encoder_decoder.decoders.recurrent import RecurrentDecoder
from nmt.encoder_decoder.decoders.greedy import GreedyDecoder


class EncDecNMT(nn.Module):

    """Encoder-Decoder network for neural machine translation."""

    def __init__(self, dict_args, inference=False):
        """
        Initialize LanguageModel.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(EncDecNMT, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.max_sent_len = dict_args["max_sent_len"]
        self.enc_hidden_dim = dict_args["enc_hidden_dim"]
        self.dec_hidden_dim = dict_args["dec_hidden_dim"]
        self.enc_vocab_size = dict_args["enc_vocab_size"]
        self.dec_vocab_size = dict_args["dec_vocab_size"]
        self.bos_idx = dict_args["bos_idx"]
        self.eos_idx = dict_args["eos_idx"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]

        # encoder
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.enc_vocab_size,
                     'hidden_size': self.enc_hidden_dim,
                     'batch_size': self.batch_size}

        if self.attention:
            self.encoder = BidirectionalEncoder(dict_args)
        else:
            self.encoder = RecurrentEncoder(dict_args)
        self.encoder.init_hidden(self.batch_size)

        # decoder
        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'word_embdim': self.word_embdim,
                     'vocab_size': self.dec_vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.dec_hidden_dim,
                     'batch_size': self.batch_size,
                     'attention': self.attention}
        self.decoder = RecurrentDecoder(dict_args)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.enc_vocab_size}
        self.source_word_embd = WordEmbeddings(dict_args)

        # inference decoder
        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'word_embdim': self.word_embdim,
                     'vocab_size': self.dec_vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.dec_hidden_dim,
                     'batch_size': self.batch_size,
                     'attention': self.attention,
                     'bos_idx': self.bos_idx,
                     'eos_idx': self.eos_idx}
        self.inference_decoder = GreedyDecoder(dict_args)

    def forward(self, source_indexseq, s_lengths,
                target_indexseq=None, t_lengths=None, inference=False):
        """Forward pass."""
        batch_size = source_indexseq.size()[0]

        self.encoder.detach_hidden(batch_size)
        source_seq_word_embds = self.source_word_embd(source_indexseq)
        source_seq_enc_states, z0 = self.encoder(
            source_seq_word_embds, s_lengths)

        if inference:
            # TODO: Implement BeamSearchDecoder
            # self.decoder = BeamSearchDecoder(dict_args)
            # raise NotImplementedError("Beam Search Decoder not implemented!")
            seq_indexes = self.inference_decoder(
                source_seq_enc_states, z0, self.decoder.state_dict())

            return seq_indexes

        else:
            log_probs = self.decoder(
                target_indexseq, t_lengths, source_seq_enc_states, z0)

        return log_probs
