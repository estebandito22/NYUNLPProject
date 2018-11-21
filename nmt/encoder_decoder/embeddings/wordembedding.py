"""PyTorch class for word embedding."""

import torch
import torch.nn as nn


class WordEmbeddings(nn.Module):

    """Class to embed words."""

    def __init__(self, dict_args):
        """
        Initialize WordEmbeddings.

        Args
            dict_args: dictionary containing the following keys:
                word_embdim: The dimension of the lookup embedding.
                vocab_size: The count of words in the data set.
                word_embeddings: Pretrained embeddings.
        """
        super(WordEmbeddings, self).__init__()

        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.vocab_size = dict_args["vocab_size"]

        self.embeddings = nn.Embedding(self.vocab_size, self.word_embdim)

        if self.word_embeddings is not None:
            if not isinstance(self.word_embeddings, torch.Tensor):
                self.word_embeddings = torch.from_numpy(self.word_embeddings)
            self.embeddings.weight = nn.Parameter(self.word_embeddings)
            self.embeddings.weight.requires_grad = False

    def forward(self, indexseq):
        """
        Forward pass.

        Args
            indexseq: A tensor of sequences of word indexes of size
                      batch_size x seqlen.
        """
        # batch_size x seqlen x embd_dim
        return self.embeddings(indexseq).permute(1, 0, 2).contiguous()
