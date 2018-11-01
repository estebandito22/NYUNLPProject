"""Script to train NMT"""

import os
import json
from collections import defaultdict
from nmt.datasets.nmt import NMTDataset
from nmt.nn.enc_dec import EncDec

inputs_dir = os.path.join(os.getcwd(), 'inputs')
train_en = os.path.join(inputs_dir, 'train.en')
train_vi = os.path.join(inputs_dir, 'train.vi')
max_sent_en = os.path.join(inputs_dir, 'max_sent_len.en')
max_sent_vi = os.path.join(inputs_dir, 'max_sent_len.vi')

files = [train_en, train_vi, max_sent_en, max_sent_vi]

data = defaultdict(list)
for file in files:
    with open(file, 'r') as f:
        r = json.load(f)
        name = file.split('/')[-1]
        data[name] = r

max_sent_len = max(data['max_sent_len.en'], data['max_sent_len.vi'])

# sent_lens = []
# for sent in data['train.vi']:
#     sent_lens += [len(sent)]
#
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# plt.hist(sent_lens, bins=100, density=True, cumulative=True)
# plt.xlim([0,100])

train_dataset = NMTDataset(data['train.vi'][:40], data['train.en'][:40], 40)
val_dataset = NMTDataset(data['train.vi'][:10], data['train.en'][:10], 40)

save_dir = os.path.join(os.getcwd(), 'outputs')

encdec = EncDec(word_embdim=10, word_embeddings=None, vocab_size=50000,
                enc_hidden_dim=10, dec_hidden_dim=10, enc_dropout=0,
                dec_dropout=0, num_layers=1, attention=True, batch_size=4,
                lr=0.01)

encdec.fit(train_dataset, val_dataset, save_dir)
