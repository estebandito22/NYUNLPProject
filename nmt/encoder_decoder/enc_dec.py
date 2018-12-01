"""PyTorch classe for encoder-decoder netowork."""

from torch import nn

from nmt.encoder_decoder.embeddings.wordembedding import WordEmbeddings

from nmt.encoder_decoder.encoders.recurrent import RecurrentEncoder
from nmt.encoder_decoder.encoders.bidirectional import BidirectionalEncoder
from nmt.encoder_decoder.encoders.convolutional import ConvolutionalEncoder

from nmt.encoder_decoder.decoders.recurrent import RecurrentDecoder
from nmt.encoder_decoder.decoders.randomteacher import RandomTeacherDecoder
from nmt.encoder_decoder.decoders.greedy import GreedyDecoder
from nmt.encoder_decoder.decoders.beamsearch import BeamDecoder
# from nmt.encoder_decoder.decoders.beamsearch import BeamDecoder


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
        self.enc_vocab_size = dict_args["enc_vocab_size"]
        self.dec_vocab_size = dict_args["dec_vocab_size"]
        self.bos_idx = dict_args["bos_idx"]
        self.eos_idx = dict_args["eos_idx"]
        self.enc_hidden_dim = dict_args["enc_hidden_dim"]
        self.dec_hidden_dim = dict_args["dec_hidden_dim"]
        self.enc_num_layers = dict_args["enc_num_layers"]
        self.dec_num_layers = dict_args["dec_num_layers"]
        self.enc_dropout = dict_args["enc_dropout"]
        self.dec_dropout = dict_args["dec_dropout"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.beam_width = dict_args["beam_width"]
        self.kernel_size = dict_args["kernel_size"]
        self.model_type = dict_args["model_type"]
        self.tf_ratio = dict_args["tf_ratio"]

        # encoder
        dict_args = {'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings[0],
                     'vocab_size': self.enc_vocab_size,
                     'hidden_size': self.enc_hidden_dim,
                     'num_layers': self.enc_num_layers,
                     'dropout': self.enc_dropout,
                     'batch_size': self.batch_size,
                     'kernel_size': self.kernel_size,
                     'model_type': self.model_type}

        if self.attention:
            self.encoder = BidirectionalEncoder(dict_args)
        else:
            self.encoder = RecurrentEncoder(dict_args)
        self.encoder.init_hidden(self.batch_size)

        if self.kernel_size > 0:
            self.encoder = ConvolutionalEncoder(dict_args)

        # decoder
        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'enc_num_layers': self.enc_num_layers,
                     'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings[1],
                     'num_layers': self.dec_num_layers,
                     'dropout': self.dec_dropout,
                     'vocab_size': self.dec_vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.dec_hidden_dim,
                     'batch_size': self.batch_size,
                     'attention': self.attention,
                     'bos_idx': self.bos_idx,
                     'eos_idx': self.eos_idx,
                     'model_type': self.model_type,
                     'tf_ratio': self.tf_ratio}
        self.decoder = RecurrentDecoder(dict_args)
        # self.decoder = RandomTeacherDecoder(dict_args)

        # inference decoder
        dict_args = {'enc_hidden_dim': self.enc_hidden_dim,
                     'enc_num_layers': self.enc_num_layers,
                     'word_embdim': self.word_embdim,
                     'word_embeddings': self.word_embeddings[1],
                     'num_layers': self.dec_num_layers,
                     'dropout': self.dec_dropout,
                     'vocab_size': self.dec_vocab_size,
                     'max_sent_len': self.max_sent_len,
                     'hidden_size': self.dec_hidden_dim,
                     'batch_size': self.batch_size,
                     'attention': self.attention,
                     'bos_idx': self.bos_idx,
                     'eos_idx': self.eos_idx,
                     'model_type': self.model_type}

        # self.inference_decoder = BeamDecoder(dict_args)
        self.inference_decoder = GreedyDecoder(dict_args)

    def forward(self, source_indexseq, s_lengths,
                target_indexseq=None, t_lengths=None, inference=False):
        """Forward pass."""
        batch_size = source_indexseq.size()[0]

        # self.encoder.detach_hidden(batch_size)
        self.encoder.init_hidden(batch_size)
        source_seq_enc_states, z0 = self.encoder(source_indexseq, s_lengths)

        if inference:
            # seq_indexes = self.inference_decoder(
            #     source_seq_enc_states, z0, self.decoder.state_dict(),
            #     self.beam_width)
            seq_indexes = self.inference_decoder(
               source_seq_enc_states, z0, self.decoder.state_dict())

            return seq_indexes

        else:
            log_probs = self.decoder(
                target_indexseq, t_lengths, source_seq_enc_states, z0)

        return log_probs
