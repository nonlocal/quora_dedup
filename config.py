import pickle

config = {
    "word2vec_size": 100,
    "word2vec_iter": 8,
    "data_path" : "./data/train.csv",
    "word2vec_min_count": 2,
    "word2vec_workers" : 4,
    "word2vec_sg" : 1,
    "word2vec_iter" : 4,
    "word2vec" : pickle.load(open("./data/wordvec.pkl", "rb")),
    "dep_map" : pickle.load(open("./data/dep_map.pkl", 'rb')),
    "tag_map" : pickle.load(open("./data/tag_map.pkl", 'rb')),
    "split_ratio" : 0.8,
    "token_only":True, #use only tokens as features, no pos, no dependency
    "extra_fc":False,
    "batch_size":128,
    "output_keep_prob":1.0,
    "n_classes" : 2,
    "n_layers" : 3, 
    "n_hidden": 128,
    "n_epochs": 16,
    "batch_size": 256,
    "learning_rate": 0.001,
    'n_epochs' : 16,
    "check_iterator_batch_sanity":True,
    "ckpt" : './data/model.ckpt',
}


if config['token_only']:
    config['n_features'] = config['word2vec_size']
else:
    config['n_features'] = config['word2vec_size']+ len(
        config['tag_map']) + len(
            config['dep_map'])


