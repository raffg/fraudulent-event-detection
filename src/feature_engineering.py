import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def main():
    pass


def feature_engineering(df):
    df['fraud'] = df['acct_type'].str.contains('fraudster')

    df = datetime(df, ['approx_payout_date', 'event_created', 'event_end',
                       'event_published', 'event_start'])
    df = hour(df, ['approx_payout_date_dt', 'event_created_dt', 'event_end_dt',
                   'event_published_dt', 'event_start_dt'])
    df = identify_empties(df, ['previous_payouts', 'payout_type'])
    df = org_blacklist(df)

    df['listed'] = df['listed'].apply(lambda x: 1 if x == 'y' else 0)
    df['user_age_90'] = df['user_age'].apply(lambda x: 1 if x >= 90 else 0)

    num_links = []
    for desc in df.description.values:
        soup = BeautifulSoup(desc, 'html')
        num_links.append(len(soup.findAll('a')))
    df['num_links'] = num_links

    df = short_dummify(df, ['email_domain', 'venue_country', 'country',
                            'currency'])

    df['total_price'] = df.ticket_types.apply(get_total_price)
    df['max_price'] = df.ticket_types.apply(get_max_price)
    df['num_tiers'] = df.ticket_types.apply(len)

    return df


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


def hour(df, columns):
    '''
    Takes a DataFrame and a list of columns with datetime and creates new
    columns with the hour of the datetime
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    for column in columns:
        df[column[:-3] + '_hour'] = df[column].dt.hour
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


def get_total_price(ticket_info):
    """Returns value of show IF it sells out.
    """
    return sum([ticket['cost']*ticket['quantity_total'] for ticket in
                ticket_info])


def get_max_price(ticket_info):
    """Returns price of most expensive ticket.
    """
    prices = [ticket['cost'] for ticket in ticket_info]
    if prices:
        return max(prices)
    return 0


if __name__ == '__main__':
    main()
