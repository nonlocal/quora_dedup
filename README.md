**Part 1 : setup the environment**   

This repo contains code for experiments done on Quora Question Classification task.

Given two text questions, the goal is to predict whether the pair intends to ask the same info or not.

To setup the training environment for this repo:
1. Install Python 3.6 or any 3+ version compatible with TensorFlow (3.7 is NOT at the moment, checkout tf page). We will assume you have anaconda.
2. Create a virtual environment name `quora_dedup` with :

    ```conda create -n quora_dedup --file requirements.txt```
3. Activate the env using:

    `source activate quora_dedup`

Now the environment is set up.


**Part 2 : data and word2vec**

Train Word2Vec from the data.
All the parameters needed are in the config.py file

run:

    python train_word2vec.py


**Part 3 : define model**

The model/feature extractor that we are going to use here is a BiLSTM followed by FC layer(s) defined in `model.py` file

**Part 4: Train the model**

Create training pipelines to train the model. Utilities define in `utils.py` and model trained in `model.py`

To change any h-params of training pipeline, make changes in config.py (instead of using argparse) and just run:
    
    python train.py


**Part 5: deploy the model**
Deploy the model and test the model by making requests to it.
See `test_api.ipynb`


**Part 6: evaluation**
Train f1 score (10 epochs) ~ `84%`

Test f1 score (10 epochs) ~ `79%`

    Evaluation.png