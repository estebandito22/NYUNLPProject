"""Script to evaluate NMT on test data."""

import os
import json
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec
from torch.utils.data import DataLoader


def main(model_dir, epoch, source_lang, reversed_in, beam_width):

    inputs_dir = os.path.join(os.getcwd(), 'inputs')
    test_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'test.en')
    test_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'test.'+source_lang)
    token2id_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'token2id.en')
    token2id_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'token2id.'+source_lang)
    id2token_en = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'id2token.en')
    id2token_sl = os.path.join(
        inputs_dir, 'iwslt-'+source_lang+'-en', 'id2token.'+source_lang)

    files = [test_en, test_sl, token2id_en, token2id_sl, id2token_en, id2token_sl]

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
        data['test.'+source_lang], data['test.en'],
        data['id2token.'+source_lang], data['id2token.en'],
        data['token2id.'+source_lang], data['token2id.en'],
        max_sent_len, reversed_in)
    val_score_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1)

    encdec = EncDec()
    encdec.load(model_dir, epoch, beam_width)
    bleu_tuple = encdec.score(val_score_loader)
    print(bleu_tuple)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("-md", "--model_dir",
                    help="Path to saved model directory.")
    ap.add_argument("-ep", "--epoch",
                    help="(int) epoch to use.")
    ap.add_argument("-sl", "--source_lang",
                    help="Source langguage of the model 'vi' or 'zh'.")
    ap.add_argument("-re", "--reversed_in", default=False, action='store_true',
                    help="Input direction of the model.")
    ap.add_argument("-bw", "--beam_width", default=1, type=int,
                    help="Beam width to set for evaluation.")

    args = vars(ap.parse_args())
    main(args["model_dir"], args["epoch"], args["source_lang"],
         args["reversed_in"], args["beam_width"])
