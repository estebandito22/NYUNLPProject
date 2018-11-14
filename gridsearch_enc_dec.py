"""Script to train NMT"""

import os
import json
import datetime
from collections import defaultdict
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec


def main(word_embdim, enc_hidden_dim, dec_hidden_dim, enc_dropout, dec_dropout,
         num_layers, attention, batch_size, lr, weight_decay, source_lang,
         num_epochs, num_searches, save_dir):

    inputs_dir = os.path.join(os.getcwd(), 'inputs')
    train_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'train.en')
    train_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'train.'+source_lang)
    dev_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'dev.en')
    dev_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'dev.'+source_lang)
    max_sent_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'max_sent_len.en')
    max_sent_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'max_sent_len.'+source_lang)
    token2id_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'token2id.en')
    token2id_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'token2id.'+source_lang)

    files = [train_en, train_sl, dev_en, dev_sl, max_sent_en,
             max_sent_sl, token2id_en, token2id_sl]

    data = defaultdict(list)
    for file in files:
        with open(file, 'r') as f:
            r = json.load(f)
            name = file.split('/')[-1]
            data[name] = r

    max_sent_len = max(
        data['max_sent_len.en'], data['max_sent_len.'+source_lang])

    enc_vocab_size = len(data['token2id.en'])
    dec_vocab_size = len(data['token2id.'+source_lang])
    vocab_size = max(enc_vocab_size, dec_vocab_size)

    train_dataset = NMTDataset(
        data['train.vi'], data['train.en'], max_sent_len)
    val_dataset = NMTDataset(
        data['dev.'+source_lang], data['dev.'+source_lang], max_sent_len)

    save_dir = os.path.join(os.getcwd(), 'outputs')

    ats = []
    ehs = []
    dhs = []
    edos = []
    ddos = []
    nls = []
    wds = []
    best_losses = []
    best_losses_train = []
    best_scores = []
    best_scores_train = []

    for _ in range(num_searches):
        eh = int(np.random.choice(enc_hidden_dim))
        dh = int(np.random.choice(dec_hidden_dim))
        edo = float(np.random.uniform(0, enc_dropout))
        ddo = float(np.random.uniform(0, dec_dropout))
        nl = int(np.random.choice(num_layers))
        wd = float(np.random.uniform(0, weight_decay))
        ed = int(np.random.choice(word_embdim))

        encdec = EncDec(word_embdim=ed,
                        word_embeddings=None,
                        vocab_size=vocab_size,
                        enc_hidden_dim=eh,
                        dec_hidden_dim=dh,
                        enc_dropout=edo,
                        dec_dropout=ddo,
                        num_layers=nl,
                        attention=attention,
                        batch_size=batch_size,
                        lr=lr,
                        weight_decay=wd,
                        num_epochs=num_epochs)

        encdec.fit(train_dataset, val_dataset, save_dir)

        ats += [attention]
        ehs += [eh]
        dhs += [dh]
        edos += [edo]
        ddos += [ddo]
        nls += [nl]
        wds += [wd]
        best_losses += [encdec.best_loss]
        best_losses_train += [encdec.best_loss_train]
        # best_scores += [encdec.best_score]
        # best_scores_train += [encdec.best_score_train]

    df = pd.DataFrame({'attention': ats,
                       'enc_hidden_dim': ehs,
                       'dec_hidden_dim': dhs,
                       'enc_dropout': edos,
                       'dec_dropout': ddos,
                       'num_layers': nls,
                       'weight_decay': wds,
                       'val_loss': best_losses,
                       'train_loss': best_losses_train,
                       'val_score': best_scores,
                       'train_score': best_scores_train})
    df.to_csv(
        os.path.join(save_dir,
                     'gridsearch_results_{}'.format(datetime.datetime.now())),
        index=False)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-ed", "--word_embdim", nargs='+', default=300, type=int,
                    help="List of ints for word embedding dimension.")
    ap.add_argument('-eh', '--enc_hidden_dim', nargs='+', default=256,
                    type=int, help='Space separated list of ints for encoder hidden dims.')
    ap.add_argument('-dh', '--dec_hidden_dim', nargs='+', default=256,
                    type=int, help='Space separated list of ints for decoder hidden dims.')
    ap.add_argument("-edo", "--enc_dropout", default=0.0, type=float,
                    help="Max encoder network dropout. NoOp if num_layers = 1.")
    ap.add_argument("-ddo", "--dec_dropout", default=0.0, type=float,
                    help="Max eecoder network dropout. NoOp if num_layers = 1.")
    ap.add_argument('-nl', '--num_layers', nargs='+', default=1,
                    type=int, help='Space separated list of ints for encoder and decoder num_layers')
    ap.add_argument("-at", "--attention", default=False, action='store_true',
                    help="Use attention in decoder.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int,
                    help="Batch size for training.")
    ap.add_argument("-lr", "--lr", default=1e-3, type=float,
                    help="Learning rate for training.")
    ap.add_argument("-wd", "--weight_decay", default=0.0, type=float,
                    help="Weight decay for training.")
    ap.add_argument("-sl", "--source_lang", default='vi',
                    help="Either 'vi' or 'zh'.")
    ap.add_argument("-ne", "--num_epochs", default=20, type=int,
                    help="Number of epochs.")
    ap.add_argument("-ns", "--num_searches", default=20, type=int,
                    help="Number of models to search over.")
    ap.add_argument("-sd", "--save_dir", default='outputs',
                    help="Save directory path.")

    args = vars(ap.parse_args())
    main(args["word_embdim"],
         args["enc_hidden_dim"],
         args["dec_hidden_dim"],
         args["enc_dropout"],
         args["dec_dropout"],
         args["num_layers"],
         args["attention"],
         args["batch_size"],
         args["lr"],
         args["weight_decay"],
         args["source_lang"],
         args["num_epochs"],
         args["num_searches"],
         args["save_dir"])
