import pandas as pd
import numpy as np
import pickle
from src.standardize import standardize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    run_model_logistic_regression()


def prepare_data():
    df = pd.read_json('data/data.json')
    y = df['acct_type'].str.contains('fraudster')

    cols = [#'acct_type',
             #'approx_payout_date',
             'body_length',
             'channels',
             #'country',
             #'currency',
             #'delivery_method', (text)
             #'description', (text)
             #'email_domain', (text)
             #'event_created',
             #'event_end',
             #'event_published',
             #'event_start',
             'fb_published',
             'gts',
             'has_analytics',
             #'has_header', (has NaNs, needs cleaning)
             'has_logo',
             'listed', 
             'name',
             'name_length',
             'num_order',
             'num_payouts',
             'object_id',
             'org_desc',
             'org_facebook',
             'org_name',
             'org_twitter',
             'payee_name',
             'payout_type',
             'previous_payouts',
             'sale_duration',
             'sale_duration2',
             'show_map',
             'ticket_types',
             'user_age',
             'user_created',
             'user_type',
             'venue_address',
             'venue_country',
             'venue_latitude',
             'venue_longitude',
             'venue_name',
             'venue_state',
             'fraud',
             'approx_payout_date_dt',
             'event_created_dt',
             'event_end_dt',
             'event_published_dt',
             'event_start_dt',
             'approx_payout_date_hour',
             'event_created_hour',
             'event_end_hour',
             'event_published_hour',
             'event_start_hour',
             'previous_payouts?',
             'payout_type?',
             'org_blacklist',
             'user_age_90',
             'num_links',
             'fraud_email_domain',
             'fraud_venue_country',
             'fraud_country',
             'fraud_currency']

    return X_train, X_test, y_train, y_test


def run_model_logistic_regression():
    X_train = pd.read_pickle('pickle/train_all_std.pkl')
    X_val = pd.read_pickle('pickle/test_all_std.pkl')
    y_train = pd.read_pickle('pickle/y_train_all_std.pkl')
    y_val = pd.read_pickle('pickle/y_test_all_std.pkl')

    feat = ['favorite_count', 'is_retweet', 'retweet_count', 'is_reply',
            'compound', 'v_negative', 'v_neutral', 'v_positive', 'anger',
            'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive',
            'sadness', 'surprise', 'trust', 'tweet_length',
            'avg_sentence_length', 'avg_word_length', 'commas',
            'semicolons', 'exclamations', 'periods', 'questions', 'quotes',
            'ellipses', 'mentions', 'hashtags', 'urls', 'is_quoted_retweet',
            'all_caps', 'tweetstorm', 'hour', 'hour_20_02', 'hour_14_20',
            'hour_08_14', 'hour_02_08', 'start_mention']

    drop = ['created_at', 'id_str', 'in_reply_to_user_id_str', 'tweetokenize',
            'text', 'pos', 'ner']

    y_train = np.array(y_train).ravel()
    y_val = np.array(y_val).ravel()

    print('All features')
    lr_all_features = lr(X_train[feat], X_val[feat], y_train, y_val)
    print()

    whole_train = X_train.drop(drop, axis=1)
    whole_val = X_val.drop(drop, axis=1)

    print('Whole model')
    lr_whole = lr(whole_train, whole_val,
                  y_train, y_val)
    print()

    top_feat = set(np.load('pickle/top_features.npz')['arr_0'][:200])
    train_feat = []
    val_feat = []
    for feat in top_feat:
        if feat in whole_train.columns:
            train_feat.append(feat)
        if feat in whole_val.columns:
            val_feat.append(feat)

    print('condensed model')
    condensed_train = whole_train[train_feat]
    condensed_val = whole_val[val_feat]

    lr_condensed = lr(condensed_train[train_feat],
                      condensed_val[val_feat],
                      y_train, y_val)

    # logistic_regression_save_pickle(lr_condensed)


def lr(X_train, X_val, y_train, y_val):
    # Logistic Regression

    model = LogisticRegression(C=.05)
    model.fit(X_train, y_train)
    predicted = model.predict(X_val)
    print('Accuracy: ', accuracy_score(y_val, predicted))
    print('Precision: ', precision_score(y_val, predicted))
    print('Recall: ', recall_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted))

    return model


def logistic_regression_save_pickle(model):
    # Save pickle file
    output = open('pickle/lr_model.pkl', 'wb')
    print('Pickle dump model')
    pickle.dump(model, output, protocol=4)
    output.close()

    return


if __name__ == '__main__':
    main()
