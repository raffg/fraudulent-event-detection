import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re


def main():
    pass


def feature_engineering(df, predict=False):
    print('Performing feature engineering')
    if not predict:
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
        soup = BeautifulSoup(desc, 'lxml')
        num_links.append(len(soup.findAll('a')))
    df['num_links'] = num_links

    df = short_dummify(df, ['email_domain', 'venue_country', 'country',
                            'currency'])

    df = dummify_nan(df, ['event_published'])
    try:
        print('Calculating total ticket price')
        df['total_price'] = df.ticket_types.apply(get_total_price)
        print('Calculating maximum ticket price')
        df['max_price'] = df.ticket_types.apply(get_max_price)
        print('Calculating the number of ticket tiers')
        df['num_tiers'] = df.ticket_types.apply(len)
    except:
        df['total_price'] = 0
        df['max_price'] = 0
        df['num_tiers'] = 1

    df['clean_desc'] = df.description.apply(clean_text)
    print()

    return df


def datetime(df, columns):
    '''
    Takes a DataFrame and a list of columns with time since the epoch and
    creates new columns with the times in datetime format
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    print('Converting datetime')
    for column in columns:
        print('   converting column ' + column)
        df[column + '_dt'] = pd.to_datetime(df[column], unit='s')
    return df


def hour(df, columns):
    '''
    Takes a DataFrame and a list of columns with datetime and creates new
    columns with the hour of the datetime
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    print('Converting hour')
    for column in columns:
        print('   converting column ' + column)
        df[column[:-3] + '_hour'] = df[column].dt.hour
    return df


def identify_empties(df, columns):
    '''
    Takes a DataFrame and a list of columns and outputs 1 if the column is
    empty and 0 otherwise
    INPUT: DataFrame, list of strings
    OUTPUT: DataFrame
    '''
    print('Identifying NaN values')
    for column in columns:
        print('   converting column ' + column)
        df[column + '?'] = df[column].apply(lambda x: 1 if len(x) > 0 else 0)
    return df


def org_blacklist(df):
    '''
    Takes a DataFrame and creates a new column with 1 if the org_name is on a
    blacklist and 0 otherwise
    INPUT: DataFrame
    OUTPUT: DataFrame
    '''
    print('Applying organization blacklist')
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
    print('Dummifying columns')
    for column in columns:
        print('   converting column ' + column)

        col_dict = {'email_domain': ['gmail.com', 'hotmail.com', 'cox.net',
                                     'hotmail.co.uk', 'yahoo.com', 'ymail.com',
                                     'inbox.com', 'zumba-perth.com',
                                     'Safe-mail.net', 'yopmail.com', 'aol.com',
                                     'diversity-church.com', 'comcast.net',
                                     'yahoo.co.uk', 'yahoo.ca',
                                     '19sieunhan.com', 'mail.com',
                                     '4asdkids.com', 'emgay.com', 'live.com',
                                     'outlook.com', 'maroclancers.com',
                                     'hotmail.fr', 'lidf.co.uk', 'gcase.org',
                                     'checker.vn', 'The2Half.com',
                                     'petlover.com', 'rocketmail.com',
                                     'aol.co.uk', 'jcclain.com',
                                     'yahoo.fr', 'live.fr', 'yahoo.com.vn',
                                     'yahoo.de', 'qip.ru', 'rock.com',
                                     'att.net', 'lmtexformula.com', 'chef.net',
                                     'ovidcapita.com', 'instructor.net',
                                     'quaychicago.com',
                                     'thinktankconsultancy.com', '9and1.biz',
                                     'GMAIL.COM', 'startupmaroc.com',
                                     'msn.com', 'consultant.com', 'hotmail.de',
                                     'ultimatewine.co.uk', 'yahoo.it',
                                     'me.com', 'safe-mail.net', 'europe.com',
                                     'ioccupied.net', 'gmx.com',
                                     'investocorp.com', 'ashfordradtech.org',
                                     'myself.com', 'inorbit.com', 'post.com',
                                     'leisurelodgebaguio.com', 'cs.com',
                                     'naworld-x.com', 'DionJordan.com',
                                     'brew-master.com', 'ravemail.com',
                                     'angelwish.org', 'hotelvenizbaguio.com',
                                     'socialworker.net', 'discofan.com',
                                     'greenrcs.com', 'outlook.fr', 'live.FR',
                                     'googlemail.com', 'toke.com',
                                     'eng.uk.com', 'outlook.de', 'mohmal.com',
                                     'yahoo.co.id', 'visichathosting.net',
                                     'catchatt.com', 'twcny.rr.com',
                                     'frontier.com', 'live.co.uk', 'clerk.com',
                                     'jobsfc.com', 'contractor.net',
                                     'primehire.co.uk', 'myway.com',
                                     'indglobal-consulting.com',
                                     'lushsaturdays.co.uk',
                                     'levyresourcesinc.com', 'in.com',
                                     'techie.com', '31and7.com', 'izzane.net',
                                     'vncall.net', 'dr.com',
                                     'gosimplysocial.com', 'freya.pw',
                                     '4u2nv-ent.com', 'accountant.com',
                                     'innovateyours.com', 'noiphuongxa.com',
                                     'anasconcept.com', 'indiabestplace.com',
                                     'live.de', 'student.framingham.edu',
                                     'hamptonmedi.com', 'hmshost.com',
                                     'execs.com', 'keromail.com',
                                     'photographer.net', 'yahoo.com.ar',
                                     'usa.com', 'clothmode.com',
                                     'cyberservices.com', 'kbzaverigroup.com',
                                     'monkeyadvert.com',
                                     'insuranceadjustersinc.com', '126.com',
                                     'nbuux.com', 'qualityservice.com',
                                     'medicalrepinsight.com',
                                     'brew-meister.com', 'blader.com',
                                     'hushmail.com', 'cdrenterprise.net',
                                     'london.com', 'fridayzonmarz.co.uk',
                                     'gawab.com', 'cannapro.com'],
                    'venue_country': ['US', '', 'GB', None, 'CA', 'AU', 'AR',
                                      'PH', 'IT', 'MA', 'ID', 'NL', 'DE', 'VN',
                                      'AE', 'FR', 'DK', 'KH', 'NA', 'KE', 'PK',
                                      'SE', 'CM', 'MX', 'DZ', 'ZA', 'RU', 'TR',
                                      'TH', 'CO', 'NG', 'OM', 'JE', 'CY',
                                      'HR'],
                    'country': ['US', '', 'GB', 'CA', 'VN', 'AU', 'MY', 'PK',
                                'MA', 'AR', 'NZ', 'CH', 'PH', 'A1', 'CI', 'ID',
                                'NL', 'DE', 'PS', 'PT', 'TR', 'NG', 'CZ', 'FR',
                                'PR', 'KH', 'JM', 'NA', 'FI', 'BG', 'GH', 'QA',
                                'SI', 'BE', 'IN', 'CM', 'RU', 'DZ', 'RO', 'IL',
                                'CN', 'RS', 'DK', 'CO', 'JE', 'HR', 'ES'],
                    'currency': ['USD', 'GBP', 'CAD', 'AUD', 'EUR', 'MXN']
                    }

        cols = col_dict[column]
        df['fraud_' + column] = df[column].apply(lambda x: 1 if x in cols
                                                 else 0)
    return df


def dummify_nan(df, columns):
    '''
    Takes a DataFrame and a list of columns and creates a new column with 1 in
    rows where the original column was NaN, otherwise 0, and drops the column
    INPUT: DataFrame, list
    OUTPUT: DataFrame
    '''
    print('Dummifying NaNs')
    for column in columns:
        print('   converting column ' + column)
        df[column + '_dummy'] = df[column].apply(lambda x: 1 if
                                                 np.isnan(x) else 0)
        df = df.drop(column, axis=1)
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


def clean_text(html_text):
    '''
    Takes a string of html code and extracts the text and replaces certain
    unicode specific characters
    Input: string
    Output: string
    '''
    soup = BeautifulSoup(html_text, 'lxml')
    c_text = soup.text
    c_text = re.sub('(\xa0)|\n', ' ', c_text)
    c_text = re.sub('\'', '', c_text)

    return c_text


if __name__ == '__main__':
    main()
