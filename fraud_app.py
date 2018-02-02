from collections import Counter
from flask import Flask, request
import numpy as np
from predict import main as predict
from sklearn.ensemble import GradientBoostingClassifier
import pickle
app = Flask(__name__)

def get_model(path):
    """Returns model from pickle fie.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def parse_message(message):
    """Returns vectorized parameters from message.
    """
    return

@app.route('/event_validation')
def api_fraud():
    model = get_model('gb_model.pkl')
    return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
