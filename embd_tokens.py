import os
import json
import numpy as np

from nmt.preprocessing.embdbuilder import EmbeddingBuilder

en = EmbeddingBuilder('/Users/stephencarrow/Documents/DS-GA 1011 Natural Language Processing and Representation/NYUNLPProject/data/elmo_en')
vi = EmbeddingBuilder('/Users/stephencarrow/Documents/DS-GA 1011 Natural Language Processing and Representation/NYUNLPProject/data/elmo_vi')
zh = EmbeddingBuilder('/Users/stephencarrow/Documents/DS-GA 1011 Natural Language Processing and Representation/NYUNLPProject/data/elmo_zh')

cwd = os.getcwd()

with open(os.path.join(cwd, 'inputs', 'iwslt-vi-en', 'id2token.en'), 'r') as f:
    vi_en_id2token_en = json.load(f)

with open(os.path.join(cwd, 'inputs', 'iwslt-vi-en', 'id2token.vi'), 'r') as f:
    vi_en_id2token_vi = json.load(f)

with open(os.path.join(cwd, 'inputs', 'iwslt-zh-en', 'id2token.en'), 'r') as f:
    zh_en_id2token_en = json.load(f)

with open(os.path.join(cwd, 'inputs', 'iwslt-zh-en', 'id2token.zh'), 'r') as f:
    zh_en_id2token_zh = json.load(f)

vi_en_embmat_en = en.build_emb_matrix(vi_en_id2token_en)
np.save(os.path.join(cwd, 'inputs', 'iwslt-vi-en', 'elbo_embds.en'),
        vi_en_embmat_en)

vi_en_embmat_vi = vi.build_emb_matrix(vi_en_id2token_vi)
np.save(os.path.join(cwd, 'inputs', 'iwslt-vi-en', 'elbo_embds.vi'),
        vi_en_embmat_vi)

zh_en_embmat_en = en.build_emb_matrix(zh_en_id2token_en)
np.save(os.path.join(cwd, 'inputs', 'iwslt-zh-en', 'elbo_embds.en'),
        zh_en_embmat_en)

zh_en_embmat_zh = zh.build_emb_matrix(zh_en_id2token_zh)
np.save(os.path.join(cwd, 'inputs', 'iwslt-zh-en', 'elbo_embds.zh'),
        zh_en_embmat_zh)
