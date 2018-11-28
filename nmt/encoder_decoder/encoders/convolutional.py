import numpy as np

from torch import nn


class ConvolutionalEncoder(nn.Module):

    def __init__(self, dict_args):
        """
        Initialize BidirectionalEncoder.

        Args
            dict_args: dictionary containing the following keys:
        """
        super(ConvolutionalEncoder, self).__init__()
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings = dict_args["word_embeddings"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.kernel_size = dict_args["kernel_size"]
        self.num_layers = dict_args["num_layers"]
        self.batch_size = dict_args["batch_size"]
        self.dropout_prob = dict_args["dropout_prob"]

        # CNN
        self.conv1 = nn.Conv1d(self.word_embdim, self.hidden_size, self.kernel_size, padding=self.kernel_size)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, self.kernel_size, padding=self.kernel_size)
        self.max_pool = nn.MaxPool1d(self.kernel_size, stride=2)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, sentence):
        out = self.conv1(sentence)
        out = self.dropout(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.drop2(out)
        out = self.relu2(out)

        out = out.permute(1, 0, 2).contiguous().view(self.batch_size, -1)
        return out