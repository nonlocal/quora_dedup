import typing
from typing import Dict

import numpy as np
import pandas as pd
import en_core_web_sm


def read_dataframe(csv_path: str, shuffle: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    if shuffle:
        for _ in range(16):
            df = df.sample(frac=1.)
    df = df.reset_index(inplace=True, drop=True)
    return df


def get_unique_questions(df: pd.DataFrame) -> Dict:
    """
    Create easy question_id :-> question_text mapping
    """
    questions = {}  # id -> question mapping

    for row in df.values:
        _, id1, id2, q1, q2, _ = row

        if id1 not in questions:
            questions[id1] = q1

        if id2 not in questions:
            questions[id2] = q2

    return questions


class SentenceIterator(object):
    def __init__(self, sentences):
        """
        Sentence Iterator for training gensim w2v
        """
        self.sentences = sentences
        # Disable heavy pipelines to improve runtime, 
        # All we need here is the tokenized text
        self.nlp = en_core_web_sm.load(disable=['ner', "tagger", "parser"])

    def __iter__(self):
        nlp = self.nlp
        for sent in self.sentences:
            # yield [x.text for x in nlp(sent)]
            # imp to lower case. Not bothered with NER type of
            #  thing where it can benefit from caps
            yield [x.text.lower() for x in nlp(sent)]
