from collections import Counter
from flask import Flask, request, json
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


# Form page to submit text
@app.route('/')
def submission_page():
    return '''
        <form action="/score" method='POST' >
            <input type="json" name="user_input" />
            <input type="submit" />
        </form>
        '''

@app.route('/score', methods = ['POST'])
def api_score():
    """Prints the prediction for the provided event information.
    """
    model = get_model('gb_model.pkl')
    data = request.form['user_input']
    prediction = predict(model, data)
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
