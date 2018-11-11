"""Script to build model inputs."""

import os
import json

from argparse import ArgumentParser

from nmt.preprocessing.inputbuilder import InputBuilder


def main(vocab_size):
    """Build the inputs."""
    cwd = os.getcwd()
    ib_vi = InputBuilder(vocab_size=vocab_size)
    ib_en = InputBuilder(vocab_size=vocab_size)

    train_vi = os.path.join(cwd, 'data', 'iwslt-vi-en', 'train.tok.vi')
    train_en = os.path.join(cwd, 'data', 'iwslt-vi-en', 'train.tok.en')
    dev_vi = os.path.join(cwd, 'data', 'iwslt-vi-en', 'dev.tok.vi')
    dev_en = os.path.join(cwd, 'data', 'iwslt-vi-en', 'dev.tok.en')
    test_vi = os.path.join(cwd, 'data', 'iwslt-vi-en', 'test.tok.vi')
    test_en = os.path.join(cwd, 'data', 'iwslt-vi-en', 'test.tok.en')

    for tok_vi, tok_en, split in \
        zip([train_vi, dev_vi, test_vi],
            [train_en, dev_en, test_en], ['train', 'dev', 'test']):

        tokens_vi = []
        with open(tok_vi, 'r') as f:
            for line in f:
                tokens_vi += [line]

        tokens_en = []
        with open(tok_en, 'r') as f:
            for line in f:
                tokens_en += [line]

        if split == 'train':
            data_vi = ib_vi.fit_transform(tokens_vi, pre_tokenized=True)
            data_en = ib_en.fit_transform(tokens_en, pre_tokenized=True)
        else:
            data_vi = ib_vi.transform(tokens_vi, pre_tokenized=True)
            data_en = ib_en.transform(tokens_en, pre_tokenized=True)

        file_names = [split, 'token2id', 'id2token', 'max_sent_len']
        files_vi = [os.path.join(cwd, 'inputs', 'iwslt-vi-en', x+'.vi')
                    for x in file_names]
        files_en = [os.path.join(cwd, 'inputs', 'iwslt-vi-en', x+'.en')
                    for x in file_names]

        for files_lang, data_lang in zip([files_vi, files_en], [data_vi, data_en]):
            for file_name, data in zip(files_lang, data_lang):
                with open(file_name, 'w') as f:
                    json.dump(data, f)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-s", "--vocab_size", default=50000, type=int,
                    help="Size of the vocabulary.")
    args = vars(ap.parse_args())
    main(args["vocab_size"])
