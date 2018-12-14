"""
Interactive script to examine attention matricies of our network.
(ATOM EDITOR REQUIRED OR PLACE IN JUPYTER NOTEBOOK).
"""

import os
import json
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
%matplotlib inline


def main(model_dir, epoch, source_lang, reversed_in, beam_width):

    inputs_dir = os.path.join(os.getcwd(), 'inputs')
    dev_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'dev.en')
    dev_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'dev.'+source_lang)
    token2id_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'token2id.en')
    token2id_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'token2id.'+source_lang)
    id2token_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'id2token.en')
    id2token_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'id2token.'+source_lang)

    files = [dev_en, dev_sl, token2id_en, token2id_sl, id2token_en, id2token_sl]

    data = defaultdict(list)
    for file in files:
        with open(file, 'r') as f:
            r = json.load(f)
            name = file.split('/')[-1]
            data[name] = r

    if source_lang == 'vi':
        max_sent_len = 49  # 95th percentile of vi + en
    elif source_lang == 'zh':
        max_sent_len = 57  # 95th percentile of zh + en
    else:
        raise ValueError("Unknown source language!")

    val_dataset = NMTDataset(
        data['dev.'+source_lang][:100], data['dev.en'][:100],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'],
        max_sent_len, reversed_in)
    val_score_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1)

    encdec = EncDec()
    encdec.load(model_dir, epoch, beam_width)
    return encdec.predict(val_score_loader)

def plot_attn(pred, ref, attn, i):
    # make xlabels
    xlab = pred.split(' ') + ['<eos>']
    ylab = ['<bos>'] + ref.split(' ') + ['<eos>']

    # filter attention on source zero padding
    attn = attn.squeeze()
    attn = attn[attn.sum(1)>0, :]
    # filter attention on pred length
    attn = attn[:, :len(xlab)]

    # make plot
    fig, ax = plt.subplots(figsize=(12,8))
    plt.pcolor(attn, cmap='binary')
    ax.set_xticklabels(xlab)
    ax.set_yticklabels(ylab)
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(xlab)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(len(ylab)))
    plt.xticks(rotation='vertical')
    plt.xlabel("Predicted Sentence")
    plt.ylabel("Source Sentence")
    plt.title("Attention Weights")
    plt.tight_layout()
    plt.savefig('vi_attn_weights_sample_{}.png'.format(i))
    plt.show()


preds, truth, ref, attn = main('./outputs_vi/ENCDEC_wed_512_we_False_evs_15319_dvs_17217_ri_False_ehd_512_dhd_1024_enl_2_dnl_2_edo_0.2_ddo_0.2_di_0.1_do_0.1_at_True_bw_1_op_sgd_lr_0.250000001_wd_0.0_cg_0.1_ls_reduce_on_plateau_ml_0.0001_mt_lstm_tf_0.95_ks_0/',
                               7, 'vi', False, 1)

i = 75
plot_attn(preds[i], ref[i], attn[i], i)
