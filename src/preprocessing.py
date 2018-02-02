import pandas as pd
from src.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def featurize(df, predict=False):
    '''
    Takes in raw dataframe and turns it into X data with features
    '''
    df = feature_engineering(df, predict)

    if not predict:
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

    X = df[cols]

    if predict:
        return X
    return X, y


def prepare_data():
    '''
    Load the data, perform feature engineering, standardize, train/test split
    '''
    df = pd.read_json('data/data.json')

    X, y = featurize(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)

    cols = list(X_train.columns)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)

    return X_train, X_test, y_train, y_test, scaler
