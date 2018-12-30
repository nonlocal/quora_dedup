import logging
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook

from model import get_model
from utils import (BatchIterator, Featurizer, make_pairs_and_labels_from_df,
                   read_dataframe)

if __name__ == "__main__":
    from config import config

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # featurizer
    featurizer = Featurizer(config['word2vec'],
                            config['tag_map'],
                            config['dep_map'],
                            config['token_only'])

    logging.info("Featurizer initialized...")

    # read data
    df = read_dataframe(config['data_path'])
    logging.info("Raw data loaded...")

    # XXX : This is the most critical part of the pipeline!
    # Tokenization is the most time consuming part of this pipeline,
    # whether we decide to use extra features or not.
    # So, in the pipeline, if out iterators are not fast enough
    # in feeding the data to the model, the training could be painfully slow!
    # To get around that, we featurize ALL THE DATA IN-MEMORY!!
    # This will save us a lot of time at the cost of RAM

    # Algo:
    # Take question_id, featurize it and store it in dict
    # for fast lookup while training

    sent_mapping = {}
    for row in tqdm_notebook(df.values):
        _, id1, id2, x, y, _ = row
        if id1 not in sent_mapping:
            # don't featurize already featurized text!
            sent_mapping[id1] = featurizer.featurize(x)
        if id2 not in sent_mapping:
            # don't featurize already featurized text!
            sent_mapping[id2] = featurizer.featurize(y)
    logging.info("Features created IN-MEMORY!")

    # make train test dataset
    logging.info("Splitting data into train-test set")
    train = df.iloc[:int(config['split_ratio']*len(df))]
    test = df.iloc[int(config['split_ratio']*len(df)):]
    logging.info("Data shape train : %s,  test: %s" %
                 (train.shape, test.shape))

    logging.info("Making labelled pairs")
    train_sentence_pairs, train_tags = make_pairs_and_labels_from_df(train)
    test_sentence_pairs, test_tags = make_pairs_and_labels_from_df(test)

    logging.info("Creating train and test batch iterators")
    training_batches = BatchIterator(
        train_sentence_pairs,
        sent_mapping,
        train_tags,
        featurizer.featurize,
        config['batch_size'],
        config['n_features']
    )
    testing_batches = BatchIterator(
        test_sentence_pairs,
        sent_mapping,
        test_tags,
        featurizer.featurize,
        config['batch_size'],
        config['n_features']
    )

    output_keep_prob = config['output_keep_prob']
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    n_layers = config['n_layers']
    n_hidden = config['n_hidden']
    learning_rate = config['learning_rate']
    n_epochs = config['n_epochs']

    logging.info("Defining model")
    tf.reset_default_graph()
    model = get_model(n_features, n_classes, n_layers, n_hidden)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    logging.info("Session defined!")

    # Use scalar summary instead?
    avg_train_f_score = []
    avg_test_f_score = []

    for ep in range(n_epochs):
        # gather batch wise f-score
        train_f_score = []
        for i, batch in enumerate(training_batches):
            features, labels, sequence_lengths = batch

            # Avoid risking OOM error due to large sequence length
            if max(sequence_lengths) > 128:
                continue

            feed = {
                model.input: features,
                model.output: labels,
                model.keep_prob: output_keep_prob,
                model.seq_len: sequence_lengths,
                model.learning_rate: learning_rate
            }
            predictions, error, _ = sess.run(
                [model.logits, model.mean_error, model.train_op],
                feed_dict=feed)

            f1 = f1_score(labels.argmax(axis=1),
                          predictions.argmax(axis=1),
                          average="macro")

            train_f_score.append(f1)
            logging.info("Train :: epoch: {}, batch : {}, error : {}, f1-score : {}".format(
                ep, i, np.round(error, 2), np.round(f1, 2)))

        test_f_score = []
        for i, batch in enumerate(testing_batches):
            features, labels, sequence_lengths = batch
            if max(sequence_lengths) > 128:
                continue
            feed = {
                model.input: features,
                model.keep_prob: output_keep_prob,
                model.seq_len: sequence_lengths
            }
            predictions = sess.run(model.logits, feed_dict=feed)
            f1 = f1_score(labels.argmax(axis=1),
                          predictions.argmax(axis=1), average="macro")
            test_f_score.append(f1)
            logging.info(
                "Test :: epoch: {}, batch : {}, f1-score : {}".format(ep, i, np.round(f1, 2)))

        avg_train_f_score.append(np.mean(train_f_score))
        avg_test_f_score.append(np.mean(test_f_score))

        print("Train batched fscore: ",
              avg_train_f_score[-1], "Test batched fscore: ", avg_test_f_score[-1])

        saver.save(sess, config['ckpt'])
        logging.info("Model checkpoint saved.")

    # FIXME : use scalar summary instead!!
    pickle.dump(avg_train_f_score, open("./data/train_avg_f1.pkl", "wb"))
    pickle.dump(avg_test_f_score, open("./data/test_avg_f1.pkl", "wb"))
    # FIXME: Use summary!!!!