"""Script to train NMT"""

import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec


def main(word_embdim, vocab_size, enc_hidden_dim,
         dec_hidden_dim, enc_dropout, dec_dropout, num_layers, attention,
         batch_size, lr, source_lang, save_dir):

    inputs_dir = os.path.join(os.getcwd(), 'inputs')
    train_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'train.en')
    train_vi = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'train.'+source_lang)
    dev_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'dev.en')
    dev_vi = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'dev.'+source_lang)
    max_sent_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'max_sent_len.en')
    max_sent_vi = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'max_sent_len.'+source_lang)

    files = [train_en, train_vi, dev_en, dev_vi, max_sent_en, max_sent_vi]

    data = defaultdict(list)
    for file in files:
        with open(file, 'r') as f:
            r = json.load(f)
            name = file.split('/')[-1]
            data[name] = r

    max_sent_len = max(
        data['max_sent_len.en'], data['max_sent_len.'+source_lang])

    train_dataset = NMTDataset(
        data['train.vi'], data['train.en'], max_sent_len)
    val_dataset = NMTDataset(
        data['dev.'+source_lang], data['dev.'+source_lang], max_sent_len)

    save_dir = os.path.join(os.getcwd(), 'outputs')

    encdec = EncDec(word_embdim=word_embdim, word_embeddings=None,
                    vocab_size=vocab_size, enc_hidden_dim=enc_hidden_dim,
                    dec_hidden_dim=dec_hidden_dim, enc_dropout=enc_dropout,
                    dec_dropout=dec_dropout, num_layers=num_layers,
                    attention=attention, batch_size=batch_size,
                    lr=lr)

    encdec.fit(train_dataset, val_dataset, save_dir)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-ed", "--word_embdim", default=300, type=int,
                    help="Word embedding dimension.")
    ap.add_argument("-vs", "--vocab_size", default=50000, type=int,
                    help="Vocabulary size of the inputs.")
    ap.add_argument("-eh", "--enc_hidden_dim", default=256, type=int,
                    help="Encoder network hidden dimension.")
    ap.add_argument("-dh", "--dec_hidden_dim", default=256, type=int,
                    help="Decoder network hidden dimension.")
    ap.add_argument("-edo", "--enc_dropout", default=0.0, type=float,
                    help="Encoder network dropout. NoOp if num_layers = 1.")
    ap.add_argument("-ddo", "--dec_dropout", default=0.0, type=float,
                    help="Decoder network dropout. NoOp if num_layers = 1.")
    ap.add_argument("-nl", "--num_layers", default=1, type=int,
                    help="Number of layers in both encoder and decoder.")
    ap.add_argument("-at", "--attention", default=False, action='store_true',
                    help="Use attention in decoder.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int,
                    help="Batch size for training.")
    ap.add_argument("-lr", "--lr", default=1e-3, type=float,
                    help="Learning rate for training.")
    ap.add_argument("-sl", "--source_lang", default='vi',
                    help="Either 'vi' or 'zh'.")
    ap.add_argument("-sd", "--save_dir", default='outputs',
                    help="Save directory path.")

    args = vars(ap.parse_args())
    main(args["word_embdim"],
         args["vocab_size"],
         args["enc_hidden_dim"],
         args["dec_hidden_dim"],
         args["enc_dropout"],
         args["dec_dropout"],
         args["num_layers"],
         args["attention"],
         args["batch_size"],
         args["lr"],
         args["source_lang"],
         args["save_dir"])
