import sys
from src.feature_engineering import *
import pickle
import json

def feature_pipeline():
    '''
    Takes in single data point and creates necessary features and returns
    array of all features used in the final model
    '''
    pass

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

    data_featurized = feature_pipeline(input_data)

    prediction = model.predict_proba(data_featurized)

    print "Percent chance of fraud: {}".format(prediction * 100)



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
