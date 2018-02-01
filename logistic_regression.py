import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score
from src.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    X_train, X_test, y_train, y_test = prepare_data()
    run_model_logistic_regression()


def prepare_data():
    df = pd.read_json('data/data.json')
    df = feature_engineering(df)

    y = df['fraud']

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
             #'name', (text)
             'name_length',
             #'num_order', (transaction)
             #'num_payouts', (transaction)
             #'object_id',
             #'org_desc', (text, Rohit doing NLP )
             #'org_facebook', (not sure what this is)
             #'org_name', (text)
             #'org_twitter', (not sure what this is)
             #'payee_name', (transaction)
             #'payout_type', (transaction)
             #'previous_payouts', (dictionaries)
             #'sale_duration', (not sure what this is)
             #'sale_duration2', (not sure what this is)
             'show_map',
             #'ticket_types', (feature engineered)
             #'user_age', (feature engineered)
             #'user_created',
             #'user_type',
             #'venue_address',
             #'venue_country',
             #'venue_latitude',
             #'venue_longitude',
             #'venue_name',
             #'venue_state',
             #'fraud', (feature engineered target)
             #'approx_payout_date_dt',
             #'event_created_dt',
             #'event_end_dt',
             #'event_published_dt',
             #'event_start_dt',
             #'approx_payout_date_hour',
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
             'fraud_currency',
             'total_price',
             'max_price',
             'num_tiers']


    X = df[cols]
    X_train, X_test, y_train, y_train = train_test_split(X,y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)




    return X_train, X_test, y_train, y_train, scaler


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
