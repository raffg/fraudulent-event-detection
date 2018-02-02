import sys
from src.feature_engineering import *
from logistic_regression import featurize
import pickle
import json

def main(model_file, input_data_file):
    '''
    Unpickles model as passed in
    Takes data point in json file, creates necessary features, and
    passes into model to make fraud prediction
    '''
    with open(model_file) as f_mod:
        model = pickle.load(f_mod)

    with open(input_data_file) as f_data:
        input_data = json.load(f_data)

    # pass data through same featurizing that training data went through
    data_featurized = featurize(pd.read_json(input_data))
    # makes prediction from model
    prediction = model.predict_proba(data_featurized)

    print "Percent chance of fraud: {}".format(prediction * 100)



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
