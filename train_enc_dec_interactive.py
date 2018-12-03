"""
Interactive script to examine gradients in the network.  Must uncomment
the plot_grad_flow() function and plot_grad_flow() calls in _train_epoch() to
genrate a gradient flow plot (ATOM EDITOR REQUIRED OR PLACE IN JUPYTER NOTEBOOK).
"""

import os
import json
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec
%matplotlib inline


def main(word_embdim, pretrained_emb, enc_hidden_dim, dec_hidden_dim,
         enc_num_layers, dec_num_layers, enc_dropout, dec_dropout, dropout_in,
         dropout_out, attention, kernel_size, beam_width, batch_size, optimize,
         lr, weight_decay, clip_grad, lr_scheduler, min_lr, reversed_in,
         source_lang, num_epochs, model_type, tf_ratio, save_dir):

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
        data['train.'+source_lang][:50], data['train.en'][:50],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'],
        max_sent_len, reversed_in)
    val_dataset = NMTDataset(
        data['dev.'+source_lang][:50], data['dev.en'][:50],
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
                    dropout_in=dropout_in,
                    dropout_out=dropout_out,
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




main(word_embdim=512, pretrained_emb=False, enc_hidden_dim=512, dec_hidden_dim=1024,
         enc_num_layers=2, dec_num_layers=2, enc_dropout=0.2, dec_dropout=0.2, dropout_in=0.1,
         dropout_out=0.1, attention=True, kernel_size=0, beam_width=1, batch_size=64, optimize='sgd',
         lr=0.25, weight_decay=0.0, clip_grad=0.1, lr_scheduler='reduce_on_plateau', min_lr=1e-4, reversed_in=False,
         source_lang='vi', num_epochs=5, model_type='lstm', tf_ratio=0.95, save_dir='outputs')
