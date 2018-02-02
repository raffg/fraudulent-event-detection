import pandas as pd
import numpy as np
from src.logistic_regression_grid_search import prepare_data
from src.ridge_grid_scan import ridge_grid_scan
from logistic_regression import lr


def main():
    X_train, y_train, scaler = prepare_data()

    # Run feature selection grid scan
    feature_list = ridge_grid_scan(X_train,
                                   np.array(y_train).ravel(),
                                   n=len(X_train.columns))

    print(feature_list)

    # np.savez('features_alphas.npz', feature_list)

    feature_list = [(x[0]) for x in list(feature_list)]

    # Save full, sorted feature list
    # np.savez('top_features.npz', feature_list)

    # Save feature list with coefficients
    model = lr(np.array(X_train),
               np.array(y_train).ravel())

    coef = model[3][0]
    features_coefs = list(zip(feature_list, coef))
    np.savez('pickle/features_coefs.npz', features_coefs)


if __name__ == '__main__':
    main()
