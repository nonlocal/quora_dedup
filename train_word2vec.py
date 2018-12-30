import logging
import pickle
import typing
from typing import Dict

import en_core_web_sm
import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from tqdm import tqdm_notebook

from utils import SentenceIterator, get_unique_questions, read_dataframe

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_w2v(config: Dict) -> bool:
    df = read_dataframe(config['data_path'])
    questions = get_unique_questions(df)
    sentences = SentenceIterator(questions.values())

    w2v_model = Word2Vec(sentences=sentences,
                         min_count=config['word2vec_min_count'],
                         workers=config['word2vec_workers'],
                         sg=config['word2vec_sg'],
                         iter=config['word2vec_iter'])
    
    w2v = {key:w2v_model.wv[key] for key in w2v_model.wv.vocab}
    pickle.dump(w2v, open(config['word2vec_path'], "wb"))
    return True

if __name__ == "__main__":
    from config import config
    train_w2v(config)