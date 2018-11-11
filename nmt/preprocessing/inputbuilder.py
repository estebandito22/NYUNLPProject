"""Class to tokenize text of multiple languages."""

import string
from collections import Counter

from tqdm import tqdm

from spacy.lang.en import English
from spacy.lang.vi import Vietnamese
from spacy.lang.zh import Chinese

class InputBuilder(object):

    """Class to transform raw text to index sequences and build vocab."""

    def __init__(self, vocab_size, language='en'):
        """Initialize InputBuilder."""
        self.vocab_size = vocab_size
        self.language = language

        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.BOS_IDX = None
        self.EOS_IDX = None

        self.token2id = None
        self.id2token = None

        self.punctuations = string.punctuation

        # initizlize the tokenizer
        if self.language == 'en':
            self.tokenizer = English().Defaults.create_tokenizer()
        elif self.language == 'vi':
            self.tokenizer = Vietnamese().Defaults.create_tokenizer()
        elif self.language == 'zh':
            self.tokenizer = Chinese().Defaults.create_tokenizer()
        else:
            raise ValueError("Unrecognized {} language!".format(self.language))

    def _lower_case_remove_punc(self, tokens):
        return [token.text.lower() for token in tokens
                if token.text not in self.punctuations
                and token.text not in ['\n']]

    def _build_vocab(self, all_tokens):
        """Build vocabulary and indexes."""
        token_counter = Counter(all_tokens)
        vocab, _ = zip(*token_counter.most_common(self.vocab_size+2))

        id2token = list(vocab)
        token2id = dict(zip(vocab, range(2, 4+len(vocab))))
        id2token = ['<pad>', '<unk>'] + id2token
        token2id['<pad>'] = self.PAD_IDX
        token2id['<unk>'] = self.UNK_IDX
        self.BOS_IDX = token2id['<bos>']
        self.EOS_IDX = token2id['<eos>']

        return token2id, id2token

    def fit_transform(self, samples, pre_tokenized=False):
        """Transform raw text samles to tokens."""
        token_dataset = []
        index_dataset = []
        all_tokens = []
        max_sent_len = 0

        if pre_tokenized:
            for sample in tqdm(samples):
                tokens = sample.split(' ')
                tokens = tokens = ['<bos>'] + tokens + ['<eos>']
                token_dataset.append(tokens)
                all_tokens += tokens

                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        else:
            for tokens in tqdm(
                    self.tokenizer.pipe(samples,
                                        batch_size=512, n_threads=1)):
                tokens = self._lower_case_remove_punc(tokens)
                tokens = ['<bos>'] + tokens + ['<eos>']
                token_dataset.append(tokens)
                all_tokens += tokens

                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        self.token2id, self.id2token = self._build_vocab(all_tokens)

        for tokens in token_dataset:
            index_dataset += [[self.token2id[token] if token in self.token2id
                               else self.UNK_IDX for token in tokens]]

        return index_dataset, self.token2id, self.id2token, max_sent_len

    def transform(self, samples, pre_tokenized=False):
        """Transform raw text samles to tokens."""
        token_dataset = []
        index_dataset = []
        max_sent_len = 0

        if pre_tokenized:
            for sample in tqdm(samples):
                tokens = sample.split(' ')
                tokens = tokens = ['<bos>'] + tokens + ['<eos>']
                token_dataset.append(tokens)

                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        else:
            for tokens in tqdm(
                    self.tokenizer.pipe(samples,
                                        batch_size=512, n_threads=1)):
                tokens = self._lower_case_remove_punc(tokens)
                tokens = ['<bos>'] + tokens + ['<eos>']
                token_dataset.append(tokens)

                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        for tokens in token_dataset:
            index_dataset += [[self.token2id[token] if token in self.token2id
                               else self.UNK_IDX for token in tokens]]

        return index_dataset, self.token2id, self.id2token, max_sent_len
