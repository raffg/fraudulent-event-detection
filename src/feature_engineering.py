import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def feature_engineering(df):
    df['fraud'] = df['acct_type'].str.contains('fraudster')

    df = datetime(df, ['approx_payout_date', 'event_created', 'event_end',
                       'event_published', 'event_start'])
    df = identify_empties(df, ['previous_payouts', 'payout_type'])
    df = org_blacklist(df)

    df['listed'] = df.replace({'listed': {'y': 1, 'n': 0}})
    df['user_age_90'] = df['user_age'].apply(lambda x: 1 if x >= 90 else 0)

    num_links = []
    for desc in df.description.values:
        soup = BeautifulSoup(desc, 'html')
        num_links.append(len(soup.findAll('a')))
    df['num_links'] = num_links

    df = short_dummify(df, ['email_domain', 'venue_country', 'country',
                            'currency'])


def datetime(df, columns):
    '''
    Takes a DataFrame and a list of columns with time since the epoch and
    creates new columns with the times in datetime format
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    for column in columns:
        df[column + '_dt'] = pd.to_datetime(df[column], unit='s')
    return df


def identify_empties(df, columns):
    '''
    Takes a DataFrame and a list of columns and outputs 1 if the column is
    empty and 0 otherwise
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    for column in columns:
        df[column + '?'] = df[column].apply(lambda x: 1 if len(x) > 0 else 0)
    return df


def org_blacklist(df):
    '''
    Takes a DataFrame and creates a new column with 1 if the org_name is on a
    blacklist and 0 otherwise
    INPUT: DataFrame
    OUTPUT: DataFrame
    '''
    blacklist = ['The London Connection',
                 "Party Starz Ent & Diverse Int'l Group",
                 'CP Enterprises',
                 'Shyone Tha MainEvent']
    df['org_blacklist'] = df['org_name'].apply(lambda x: 1 if x in blacklist
                                               else 0)
    return df


def short_dummify(df, columns):
    '''
    Takes a DataFrame and a list of columns and outputs a 1 if the value has
    been associated with fraud and 0 otherwise
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    for column in columns:
        df['fraud_' + column] = df[column].apply(lambda x: 1 if x in
                                                 list(df[df['fraud']]
                                                        [column].unique())
                                                 else 0)
    return df
