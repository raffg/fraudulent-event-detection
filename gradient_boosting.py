import pandas as pd
import numpy as np
import pickle
from src.preprocessing import featurize, prepare_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score
from src.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    run_model_gradient_boosting(X_train, X_test, y_train, y_test)


def run_model_gradient_boosting(X_train, X_test, y_train, y_test):
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    print('Running Gradient Boosting')
    model = gb(X_train, X_test, y_train, y_test)
    print()

    # logistic_regression_save_pickle(model)


def gb(X_train, X_test, y_train, y_test):
    # Gradient Boosting

    model = GradientBoostingClassifier(loss='deviance',
                                       learning_rate=.5,
                                       n_estimators=100,
                                       max_depth=3,
                                       min_samples_split=2,
                                       min_samples_leaf=2,
                                       max_features='auto')
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))

    return model


def gradient_boosting_save_pickle(model):
    # Save pickle file
    output = open('pickle/gb_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
