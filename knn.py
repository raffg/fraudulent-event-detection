import pandas as pd
import numpy as np
import pickle
from src.preprocessing import featurize, prepare_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score
from src.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    run_model_knn(X_train, X_test, y_train, y_test)

    # with open('scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f, protocol=4)


def run_model_knn(X_train, X_test, y_train, y_test):
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    print('Running KNN')
    model = knn(X_train, X_test, y_train, y_test)
    print()

    knn_save_pickle(model)


def knn(X_train, X_test, y_train, y_test):
    # K Nearest Neighbors

    model = KNeighborsClassifier(n_neighbors=4, weights='distance')
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('Recall: ', recall_score(y_test, predicted))
    print('F1 score: ', f1_score(y_test, predicted))

    return model


def knn_save_pickle(model):
    # Save pickle file
    output = open('knn_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
