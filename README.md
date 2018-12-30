This repo contains code for experiemtns done on Quora Question Classification task.

Given two text questions, the goal is to predict whether the pair intends to ask the same info or not.

To setup the training environment for this repo:
1. Install Python 3.6 or any 3+ version compatible with TensorFlow (3.7 is NOT at the moment, checkout tf page). We will assume you have anaconda.
2. Create a virtual environment name `quora_dedup` with :

    ```conda create -n quora_dedup --file requirements.txt```
3. Activate the env using:

    `source activate quora_dedup`

Now the environment is set up.