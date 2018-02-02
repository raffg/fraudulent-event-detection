import pandas as pd
import numpy as np
from src.preprocessing import featurize, prepare_data
from logistic_regression import lr
from random_forest import rf
from gradient_boosting import gb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    results = run_model_ensemble(X_train, X_test, y_train, y_test)
    X_train = pd.DataFrame({'Logistic Regression': results[0],
                            'Random Forest': results[1],
                            'Gradient Boosting': results[2]})

    y_train = y_train.reset_index(drop=True)
    temp = pd.concat([X_train, y_train], axis=1)
    X_train = temp[['Gradient Boosting',
                    'Logistic Regression',
                    'Random Forest']]
    y_train = temp['fraud']

    X_train['Majority Vote'] = 0
    X_train['Majority Vote'] = X_train.apply(majority, axis=1)

    # result = decision_tree_grid_search(X_train, y_train)
    # print(result.best_params_, result.best_score_)

    ensemble = decision_tree(X_train, y_train)
    print('ensemble trained')

    # test = pd.DataFrame({'Logistic Regression': X_test,
    #                      'Random Forest': X_test,
    #                      'Gradient Boosting': X_test})
    # print('test complete')
    #
    # test['Majority Vote'] = 0
    # test['Majority Vote'] = test.apply(majority, axis=1)
    # print('test majority complete')

    test_results = ensemble_test_results(ensemble, X_test, y_test)

    # ensemble_save_pickle(model)


def run_model_ensemble(X_train, X_test, y_train, y_test):
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    print('Training Logistic Regression')
    model_lr = lr(X_train, X_test, y_train, y_test)
    results_lr = model_lr.predict(X_train)
    print()

    print('Training Random Forest')
    model_rf = rf(X_train, X_test, y_train, y_test)
    results_rf = model_rf.predict(X_train)
    print()

    print('Training Gradient Boosting')
    model_gb = gb(X_train, X_test, y_train, y_test)
    results_gb = model_gb.predict(X_train)
    print()

    return results_lr, results_rf, results_gb, model_lr, model_rf, model_gb


def majority(row):
    val = 1 if (row['Logistic Regression'] +
                row['Random Forest'] +
                row['Gradient Boosting']) > 1 else 0
    return val


def decision_tree(X, y):
    # Basic decision tree for ensemble
    print('Training Decision Tree')

    kfold = KFold(n_splits=10)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kfold.split(X):
        model = DecisionTreeClassifier(criterion='gini',
                                       max_depth=4,
                                       max_features='sqrt',
                                       min_samples_leaf=3,
                                       min_samples_split=2,
                                       min_weight_fraction_leaf=0.0,
                                       splitter='random')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_true = y_test
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))
        f1_scores.append(f1_score(y_true, y_predict))

    accuracy = np.average(accuracies)
    precision = np.average(precisions)
    recall = np.average(recalls)
    f1 = np.average(f1_scores)

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1)

    return model


def decision_tree_grid_search(X, y):
    parameters = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_depth': [None, 2, 3, 4, 5, 6],
                  'min_samples_split': [2, 3, 4],
                  'min_samples_leaf': [1, 2, 3],
                  'min_weight_fraction_leaf': [0., .001, .01, .1, .25],
                  'max_features': [None, 'sqrt', 'log2']}

    dt = DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters, cv=10, verbose=True)
    clf.fit(X, y)

    return clf


def ensemble_test_results(model, X_test, y_test):
    y_predict = model.predict(X_test)

    print()
    print('Accuracy: ', accuracy_score(y_test, y_predict))
    print('Precision: ', precision_score(y_test, y_predict))
    print('Recall: ', recall_score(y_test, y_predict))
    print('F1 score: ', f1_score(y_test, y_predict))


def ensemble_save_pickle(model):
    # Save pickle file
    output = open('ensemble_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
