"""PyTorch classe for encoder-decoder netowork."""

from torch import nn

from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings

from nmt.encoder_decoder.encoders.recurrent import RecurrentEncoder
from nmt.encoder_decoder.encoders.bidirectional import BidirectionalEncoder

from nmt.encoder_decoder.decoders.recurrent import RecurrentDecoder


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
        self.enc_dropout = dict_args["enc_dropout"]
        self.dec_dropout = dict_args["dec_dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.num_layers = dict_args["num_layers"]

        # encoder
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'hidden_size': self.enc_hidden_dim,
                     'num_layers': self.num_layers,
                     'dropout': self.enc_dropout,
                     'batch_size': self.batch_size}

        if self.attention:
            self.encoder = BidirectionalEncoder(dict_args)
        else:
            self.encoder = RecurrentEncoder(dict_args)
        self.encoder.init_hidden(self.batch_size)

        # decoder
        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.dec_hidden_dim,
                     'dropout': self.dec_dropout,
                     'batch_size': self.batch_size,
                     'attention': self.attention}
        self.decoder = RecurrentDecoder(dict_args)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size}
        self.source_word_embd = WordEmbeddings(dict_args)

        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size}
        self.target_word_embd = WordEmbeddings(dict_args)

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
            raise NotImplementedError("Beam Search Decoder not implemented!")
        else:
            target_seq_word_embds = self.source_word_embd(target_indexseq)
            log_probs = self.decoder(
                target_seq_word_embds, t_lengths, source_seq_enc_states, z0)

        return log_probs
