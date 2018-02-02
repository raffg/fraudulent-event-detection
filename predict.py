import sys
from src.feature_engineering import *
from logistic_regression import featurize
from gradient_boosting import *
import pickle
import json

def main(model, data):
    '''
    Unpickles model as passed in
    Takes data point in json file, creates necessary features, and
    passes into model to make fraud prediction
    '''


    with open(model, 'rb') as f_mod:
        model = pickle.load(f_mod)

    # pass data through same featurizing that training data went through
    df = pd.read_json(data)
    data_featurized = featurize(df)[0]

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    data_scaled = scaler.fit_transform(data_featurized)

    # makes prediction from model
    for row in data_scaled:

        prediction = model.predict_proba(row.reshape(1,-1))

        print("Percent chance of fraud: {}".format(prediction * 100))



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
