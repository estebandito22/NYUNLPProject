"""Script to train NMT"""

import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec


def main(word_embdim, enc_hidden_dim, dec_hidden_dim, attention, batch_size,
         lr, weight_decay, source_lang, num_epochs, save_dir):

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
    id2token_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'id2token.en')
    id2token_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'id2token.'+source_lang)

    files = [train_en, train_sl, dev_en, dev_sl, max_sent_en,
             max_sent_sl, token2id_en, token2id_sl, id2token_en, id2token_sl]

    data = defaultdict(list)
    for file in files:
        with open(file, 'r') as f:
            r = json.load(f)
            name = file.split('/')[-1]
            data[name] = r

    # max_sent_len = max(
    #     data['max_sent_len.en'], data['max_sent_len.'+source_lang])

    if source_lang == 'vi':
        max_sent_len = 49  # 95th percentile of vi + en
    elif source_lang == 'zh':
        max_sent_len = 57  # 95th percentile of zh + en
    else:
        raise ValueError("Unknown source language!")

    enc_vocab_size = len(data['token2id.'+source_lang])
    dec_vocab_size = len(data['token2id.en'])

    bos_idx = data['token2id.en']['<bos>']
    eos_idx = data['token2id.en']['<eos>']
    pad_idx = data['token2id.en']['<pad>']

    train_dataset = NMTDataset(
        data['train.'+source_lang], data['train.en'],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'], max_sent_len)
    val_dataset = NMTDataset(
        data['dev.'+source_lang], data['dev.en'],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'], max_sent_len)

    encdec = EncDec(word_embdim=word_embdim,
                    word_embeddings=None,
                    enc_vocab_size=enc_vocab_size,
                    dec_vocab_size=dec_vocab_size,
                    bos_idx=bos_idx,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    enc_hidden_dim=enc_hidden_dim,
                    dec_hidden_dim=dec_hidden_dim,
                    attention=attention,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    num_epochs=num_epochs)

    encdec.fit(train_dataset, val_dataset, save_dir)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-ed", "--word_embdim", default=300, type=int,
                    help="Word embedding dimension.")
    ap.add_argument("-eh", "--enc_hidden_dim", default=256, type=int,
                    help="Encoder network hidden dimension.")
    ap.add_argument("-dh", "--dec_hidden_dim", default=256, type=int,
                    help="Decoder network hidden dimension.")
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
    ap.add_argument("-sd", "--save_dir", default='outputs',
                    help="Save directory path.")

    args = vars(ap.parse_args())
    main(args["word_embdim"],
         args["enc_hidden_dim"],
         args["dec_hidden_dim"],
         args["attention"],
         args["batch_size"],
         args["lr"],
         args["weight_decay"],
         args["source_lang"],
         args["num_epochs"],
         args["save_dir"])
