"""Script to train NMT"""

import os
import json
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec


def main(word_embdim, pretrained_emb, enc_hidden_dim, dec_hidden_dim,
         enc_num_layers, dec_num_layers, enc_dropout, dec_dropout, attention,
         kernel_size, beam_width, batch_size, optimize, lr, weight_decay, clip_grad,
         lr_scheduler, min_lr, reversed_in, source_lang,
         num_epochs, model_type, tf_ratio, save_dir):


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
    word_embds_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'elbo_embds.en.npy')
    word_embds_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'elbo_embds.'+source_lang+'.npy')

    files = [train_en, train_sl, dev_en, dev_sl, max_sent_en,
             max_sent_sl, token2id_en, token2id_sl, id2token_en, id2token_sl]

    data = defaultdict(list)
    for file in files:
        with open(file, 'r') as f:
            r = json.load(f)
            name = file.split('/')[-1]
            data[name] = r

    if kernel_size > 0 and attention:
        raise ValueError("Attention not implemented with convolutional encoder!")

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

    # must be for target language
    bos_idx = data['token2id.en']['<bos>']
    eos_idx = data['token2id.en']['<eos>']
    pad_idx = data['token2id.en']['<pad>']

    # pretrained embeddings
    if pretrained_emb:
        for file in [word_embds_en, word_embds_sl]:
            r = np.load(file)
            name = file.split('/')[-1]
            data[name] = r

        word_embeddings = (data['elbo_embds.'+source_lang+'.npy'],
                           data['elbo_embds.en.npy'])
    else:
        word_embeddings = (None, None)

    train_dataset = NMTDataset(
        data['train.'+source_lang], data['train.en'],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'],
        max_sent_len, reversed_in)
    val_dataset = NMTDataset(
        data['dev.'+source_lang], data['dev.en'],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'],
        max_sent_len, reversed_in)

    encdec = EncDec(word_embdim=word_embdim,
                    word_embeddings=word_embeddings,
                    enc_vocab_size=enc_vocab_size,
                    dec_vocab_size=dec_vocab_size,
                    bos_idx=bos_idx,
                    eos_idx=eos_idx,
                    pad_idx=pad_idx,
                    enc_hidden_dim=enc_hidden_dim,
                    dec_hidden_dim=dec_hidden_dim,
                    enc_num_layers=enc_num_layers,
                    dec_num_layers=dec_num_layers,
                    enc_dropout=enc_dropout,
                    dec_dropout=dec_dropout,
                    kernel_size=kernel_size,
                    attention=attention,
                    beam_width=beam_width,
                    batch_size=batch_size,
                    optimize=optimize,
                    lr=lr,
                    weight_decay=weight_decay,
                    clip_grad=clip_grad,
                    lr_scheduler=lr_scheduler,
                    min_lr=min_lr,
                    num_epochs=num_epochs,
                    model_type=model_type,
                    tf_ratio=tf_ratio)

    encdec.fit(train_dataset, val_dataset, save_dir)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-ed", "--word_embdim", default=300, type=int,
                    help="Word embedding dimension.")
    ap.add_argument("-pw", "--pretrained_emb", default=False,
                    action='store_true', help="Use pretrained word embedding.")
    ap.add_argument("-eh", "--enc_hidden_dim", default=256, type=int,
                    help="Encoder network hidden dimension.")
    ap.add_argument("-dh", "--dec_hidden_dim", default=256, type=int,
                    help="Decoder network hidden dimension.")
    ap.add_argument("-enl", "--enc_num_layers", default=1, type=int,
                    help="Number of layers in encoder.")
    ap.add_argument("-dnl", "--dec_num_layers", default=1, type=int,
                    help="Number of layers in decoder.")
    ap.add_argument("-edo", "--enc_dropout", default=0.0, type=float,
                    help="Dropout in encoder.  NoOp if num_layers=1.")
    ap.add_argument("-ddo", "--dec_dropout", default=0.0, type=float,
                    help="Dropout in decoder.  NoOp if num_layers=1.")
    ap.add_argument("-ke", "--kernel_size", default=0, type=int,
                    help="Use a convolutional encoder instead of RNN with specified kernel size.")
    ap.add_argument("-at", "--attention", default=False, action='store_true',
                    help="Use attention in decoder.")
    ap.add_argument("-bw", "--beam_width", default=1, type=int,
                    help="Beam width for beam search decoder.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int,
                    help="Batch size for training.")
    ap.add_argument("-op", "--optimize", default='adam',
                    help="Optimizier, 'adam' or 'sgd'.")
    ap.add_argument("-lr", "--lr", default=0.25, type=float,
                    help="Learning rate for training.")
    ap.add_argument("-wd", "--weight_decay", default=0.0, type=float,
                    help="Weight decay for training.")
    ap.add_argument("-cg", "--clip_grad", default=0.1, type=float,
                    help="Clip grad norm to value.  No clipping if 0.")
    ap.add_argument("-ls", "--lr_scheduler", default='fixed',
                    help="Learning rate scheduler. 'fixed', 'reduce_on_plateau' or 'multi_step_lr'.")
    ap.add_argument("-ml", "--min_lr", default=1e-4, type=float,
                    help="Minimum learning rate for LR scheduler.")
    ap.add_argument("-re", "--reversed_in", default=False, action='store_true',
                    help="Reverse input sentence direction.")
    ap.add_argument("-sl", "--source_lang", default='vi',
                    help="Either 'vi' or 'zh'.")
    ap.add_argument("-ne", "--num_epochs", default=20, type=int,
                    help="Number of epochs.")
    ap.add_argument("-sd", "--save_dir", default='outputs',
                    help="Save directory path.")
    ap.add_argument("-mt", "--model_type", default='lstm',
                    help="Model type to use.  'lstm' or 'gru'.")
    ap.add_argument("-tf", "--tf_ratio", default=0.5, type=float,
                    help="Teacher forcing ratio.")

    args = vars(ap.parse_args())
    main(args["word_embdim"],
         args["pretrained_emb"],
         args["enc_hidden_dim"],
         args["dec_hidden_dim"],
         args["enc_num_layers"],
         args["dec_num_layers"],
         args["enc_dropout"],
         args["dec_dropout"],
         args["attention"],
         args["kernel_size"],
         args["beam_width"],
         args["batch_size"],
         args["optimize"],
         args["lr"],
         args["weight_decay"],
         args["clip_grad"],
         args["lr_scheduler"],
         args["min_lr"],
         args["reversed_in"],
         args["source_lang"],
         args["num_epochs"],
         args["model_type"],
         args["tf_ratio"],
         args["save_dir"])
