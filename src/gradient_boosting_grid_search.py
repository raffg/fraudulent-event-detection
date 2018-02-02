import pandas as pd
import numpy as np
from src.feature_engineering import feature_engineering
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def main():
    X_train, y_train, scaler = prepare_data()
    result = gb_grid_search(np.array(X_train), np.array(y_train).ravel())
    print(result.best_params_, result.best_score_)


def gb_grid_search(X, y):
    parameters = {'loss': ['deviance', 'exponential'],
                  'learning_rate': [.1, .5, 1, 1.5, 10],
                  'n_estimators': [10, 50, 100, 150, 200],
                  'max_depth': [2, 3, 4],
                  'min_samples_split': [2, 3],
                  'min_samples_leaf': [1, 2],
                  'max_features': ['auto', 'sqrt', 'log2']}

    gb = GradientBoostingClassifier()
    clf = GridSearchCV(gb, parameters, scoring='recall', cv=10, verbose=True)
    clf.fit(X, y)

    return clf


def prepare_data():
    '''
    Load the data, perform feature engineering, standardize, train/test split
    '''
    df = pd.read_json('data/data.json')
    df = feature_engineering(df)
    df = df.dropna(subset=['event_published'])

    y = df['fraud']
    X = df.drop('fraud', axis=1)

    cols = [  # 'acct_type',
            # 'approx_payout_date',
            'body_length',
            'channels',
            # 'country',
            # 'currency',
            # 'delivery_method', (text)
            # 'description', (text)
            # 'email_domain', (text)
            # 'event_created',
            # 'event_end',
            # 'event_published',
            # 'event_start',
            'fb_published',
            # 'gts',
            'has_analytics',
            # 'has_header', (has NaNs, needs cleaning)
            'has_logo',
            'listed',
            # 'name', (text)
            'name_length',
            # 'num_order', (transaction)
            # 'num_payouts', (transaction)
            # 'object_id',
            # 'org_desc', (text, Rohit doing NLP )
            # 'org_facebook', (not sure what this is)
            # 'org_name', (text)
            # 'org_twitter', (not sure what this is)
            # 'payee_name', (transaction)
            # 'payout_type', (transaction)
            # 'previous_payouts', (dictionaries)
            # 'sale_duration', (not sure what this is)
            # 'sale_duration2', (not sure what this is)
            'show_map',
            # 'ticket_types', (feature engineered)
            # 'user_age', (feature engineered)
            # 'user_created',
            # 'user_type',
            # 'venue_address',
            # 'venue_country',
            # 'venue_latitude',
            # 'venue_longitude',
            # 'venue_name',
            # 'venue_state',
            # 'fraud', (feature engineered target)
            'event_published_dummy',
            # 'approx_payout_date_dt',
            # 'event_created_dt',
            # 'event_end_dt',
            # 'event_start_dt',
            # 'approx_payout_date_hour',
            'event_created_hour',
            'event_end_hour',
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

    X_train = df[cols]
    y_train = y

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)

    return X_train, y_train, scaler


if __name__ == '__main__':
    main()
