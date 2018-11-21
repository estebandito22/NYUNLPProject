import numpy as np
from elmoformanylangs import Embedder


class EmbeddingBuilder(object):

    """Class to build embedding matrix with pre-trained ELMo embeddings."""

    def __init__(self, model_dir):
        """Initialize EmbeddingBuilder."""
        assert isinstance(model_dir, str), "Invalid model directory provided."

        self.model_dir = model_dir
        self.embedder = Embedder(self.model_dir)

    def build_emb_matrix(self, id2token):
        """
        Build and embedding matrix for a pre-tokenized vocabulary.

        Args
            id2token : list, list of tokens with index corresponding to ids.
        """
        embd_matrix = np.concatenate(self.embedder.sents2elmo([id2token]))
        return embd_matrix
