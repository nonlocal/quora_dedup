import numpy as np
import tensorflow as tf

# Define models here
from config import config
from utils import Featurizer
from utils import ModelAPI

# from flask import Flask, jsonify, request,make_response
# from flask_cors import CORS


featurize = Featurizer(config['word2vec'],
                       config['tag_map'],
                       config['dep_map'],
                       config['token_only']).featurize

model = ModelAPI(featurize=featurize, config=config)

app = Flask(__name__)
CORS(app)


@app.route('/get_my_predictions', methods=['POST'])
def get_predictions():
    """
    expected format of json: {
        "text_pair": [text1, text2]
    }
    """

    # print(request.json)
    text_pair = request.json.get('text_pair')
    pred = model.predict(text_pair=text_pair)
    return jsonify({"is_duplicate": pred})

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', threaded=True)