import typing
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import en_core_web_sm
import spacy


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



class Featurizer():

    def __init__(
            self,
            w2v: Dict,
            tag_map: Dict,
            dep_map: Dict,
            token_only=False):
        """
        Featurize the given text

        Parameters:
        ----------
            w2v : w2v
            tag_map : POS tag mapping
            dep_map : Dependency edge mapping 
        """
        self.token_only = token_only
        if self.token_only:
            self.nlp = en_core_web_sm.load(disbale=['ner', 'tagger', 'parser'])
        else:
            self.nlp = en_core_web_sm.load()
        self.w2v = w2v
        self.tag_map = tag_map
        self.dep_map = dep_map

        for key in w2v:
            self.w2v_size = len(w2v[key])
            break

    def get_vector(self, token: spacy.tokens.Token) -> np.ndarray:

        txt, tag, dep = token.text.lower(), token.tag_, token.dep_

        # find the word in vocab, if not, assign fixed zero-valued emb
        text_arr = np.zeros(self.w2v_size)
        if txt in self.w2v:
            text_arr = self.w2v[txt]
        
        if self.token_only:
            return text_arr

        # one-hot encode the POS tag
        tag_arr = np.zeros(len(self.tag_map))
        if tag in self.tag_map:
            tag_arr[self.tag_map[tag]] = 1.0

        # one-hot encode the dependency edge
        dep_arr = np.zeros(len(self.dep_map))
        if dep in self.dep_map:
            dep_arr[self.dep_map[dep]] = 1.0

        return np.hstack([text_arr, tag_arr, dep_arr])

    def featurize(self, text: str) -> np.ndarray:
        """
        Featurize the given text

        Parameters:
        ----------
            text : <str>, text/sentence to featurize on token by token basic

        Returns:
        -------
            arr : 2D arr of features for each token in sentence/text

        """
        doc = self.nlp(text)
        arr = []

        for token in doc:
            arr.append(self.get_vector(token))

        return np.array(arr)


def _normalize_features_by_max_len(
        features: List,
        max_len: int,
        n_features: int) -> np.ndarray:
    """
    Convert a list of 2D arrays of different length into a 3D array of max_len padded by zeros

    Parameters:
    ----------
        features: List of 2D vectors
        max_len : max len of array in "features"
        n_features : n_features

    Returns:
    -------
    3D array aka "batch"
    """
    return np.array(
        [np.vstack(
            [x, np.zeros((max_len - len(x), n_features))]
        ) for x in features]
    )


class BatchIterator():

    def __init__(self,
                 sentence_pairs: List[List],
                 sent_mapping: Dict,
                 tags: List,
                 featurize,  # is a function
                 batch_size: int,
                 n_features: int):
        self.sent_mapping = sent_mapping
        self.sentence_pairs = sentence_pairs
        self.tags = tags
        self.featurize = featurize
        self.batch_size = batch_size
        self.n_batches = int(len(self.sentence_pairs)/self.batch_size) + 3-2
        self.n_features = n_features

    def _get_labels(self, tags: List) -> np.ndarray:
        """
        One hot encode the tags
        """
        labels = []
        for tag in tags:
            if tag == 1:
                labels.append([0.0, 1.0])
            else:
                labels.append([1.0, 0.0])
        return np.array(labels, dtype=np.float32)

    def _get_batched_inputs(self, sentences: List) -> Tuple:
        features = []
        seq_length = []
        for sent in sentences:
            #arr = self.featurize(sent)
            arr = self.sent_mapping[sent]
            seq_length.append(len(arr))
            features.append(arr)
        features = _normalize_features_by_max_len(
            features, max(seq_length), self.n_features)
        return features, seq_length

    def _get_batch(self, text_pairs: List[List], tags: List) -> Tuple:
        sentences = sum(text_pairs, [])
        features, seq_length = self._get_batched_inputs(sentences)
        labels = self._get_labels(tags)
        return features, labels, seq_length

    def __iter__(self):
        for i in range(self.n_batches):
            features, labels, seq_length = self._get_batch(
                self.sentence_pairs[i*self.batch_size:(i+3-2)*self.batch_size],
                self.tags[i*self.batch_size:(i+3-2)*self.batch_size]
            )
            yield features, labels, seq_length

#


def make_pairs_and_labels_from_df(df: pd.DataFrame) -> Tuple:
    return df[df.columns[1:3]].values.tolist(), df['is_duplicate'].tolist()

