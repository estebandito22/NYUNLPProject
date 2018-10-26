"""Script to build model inputs."""

import os
import json

from argparse import ArgumentParser

from nmt.preprocessing.inputbuilder import InputBuilder


def main(vocab_size):
    """Build the inputs."""
    cwd = os.getcwd()
    ib_vi = InputBuilder(vocab_size=vocab_size, language='vi')
    ib_en = InputBuilder(vocab_size=vocab_size, language='en')

    train_vi = os.path.join(cwd, 'data', 'train.vi')
    train_en = os.path.join(cwd, 'data', 'train.en')

    tokens_vi = []
    with open(train_vi, 'r') as f:
        for line in f:
            tokens_vi += [line]

    tokens_en = []
    with open(train_en, 'r') as f:
        for line in f:
            tokens_en += [line]

    data_vi = ib_vi.fit_transform(tokens_vi)
    data_en = ib_en.fit_transform(tokens_en)

    file_names = ['train', 'token2id', 'id2token', 'max_sent_len']
    files_vi = [os.path.join(cwd, 'inputs', x+'.vi') for x in file_names]
    files_en = [os.path.join(cwd, 'inputs', x+'.en') for x in file_names]

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
