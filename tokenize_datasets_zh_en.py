"""Script to build model inputs."""

import os
import json

from argparse import ArgumentParser

from nmt.preprocessing.inputbuilder import InputBuilder


def main(vocab_size):
    """Build the inputs."""
    cwd = os.getcwd()
    ib_zh = InputBuilder(vocab_size=vocab_size)
    ib_en = InputBuilder(vocab_size=vocab_size)

    train_zh = os.path.join(cwd, 'data', 'iwslt-zh-en', 'train.tok.zh')
    train_en = os.path.join(cwd, 'data', 'iwslt-zh-en', 'train.tok.en')
    dev_zh = os.path.join(cwd, 'data', 'iwslt-zh-en', 'dev.tok.zh')
    dev_en = os.path.join(cwd, 'data', 'iwslt-zh-en', 'dev.tok.en')
    test_zh = os.path.join(cwd, 'data', 'iwslt-zh-en', 'test.tok.zh')
    test_en = os.path.join(cwd, 'data', 'iwslt-zh-en', 'test.tok.en')

    for tok_zh, tok_en, split in \
        zip([train_zh, dev_zh, test_zh],
            [train_en, dev_en, test_en], ['train', 'dev', 'test']):

        tokens_zh = []
        with open(tok_zh, 'r') as f:
            for line in f:
                tokens_zh += [line]

        tokens_en = []
        with open(tok_en, 'r') as f:
            for line in f:
                tokens_en += [line]

        if split == 'train':
            data_zh = ib_zh.fit_transform(tokens_zh, pre_tokenized=True)
            data_en = ib_en.fit_transform(tokens_en, pre_tokenized=True)
        else:
            data_zh = ib_zh.transform(tokens_zh, pre_tokenized=True)
            data_en = ib_en.transform(tokens_en, pre_tokenized=True)

        file_names = [split, 'token2id', 'id2token', 'max_sent_len']
        files_zh = [os.path.join(cwd, 'inputs', 'iwslt-zh-en', x+'.zh')
                    for x in file_names]
        files_en = [os.path.join(cwd, 'inputs', 'iwslt-zh-en', x+'.en')
                    for x in file_names]

        for files_lang, data_lang in zip([files_zh, files_en], [data_zh, data_en]):
            for file_name, data in zip(files_lang, data_lang):
                with open(file_name, 'w') as f:
                    json.dump(data, f)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-s", "--vocab_size", default=50000, type=int,
                    help="Size of the vocabulary.")
    args = vars(ap.parse_args())
    main(args["vocab_size"])
