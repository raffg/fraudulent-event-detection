import pandas as pd
import numpy as np
import pickle
from src.preprocessing import featurize, prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score
from src.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    run_model_random_forest(X_train, X_test, y_train, y_test)


def run_model_random_forest(X_train, X_test, y_train, y_test):
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    print('Running Random Forest')
    model = rf(X_train, X_test, y_train, y_test)
    print()

    # lrandom_forest_save_pickle(lr_condensed)


def rf(X_train, X_test, y_train, y_test):
    # Random Forest

    model = RandomForestClassifier(max_depth=None,
                                   max_features='sqrt',
                                   min_samples_leaf=1,
                                   min_samples_split=2,
                                   n_estimators=1000,
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))

    return model


def random_forest_save_pickle(model):
    # Save pickle file
    output = open('pickle/rf_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
