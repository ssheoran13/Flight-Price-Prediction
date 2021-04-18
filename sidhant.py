import pandas as pd
from datetime import datetime, timedelta
import os.path


def combine_input_files(path_to_dir='./data_scraper/Data'):
    ''' 
    Return combined data set for all files

    Input: path to directory
    '''
    startDate = datetime(2020, 9, 28)
    df = pd.DataFrame()
    num_files = len([f for f in os.listdir(path_to_dir)
                     if os.path.isfile(os.path.join(path_to_dir, f))])
    for i in range(num_files):
        temp = pd.read_csv('{}/FlightsData_{}.csv'.format(path_to_dir,
                                                          (startDate+timedelta(i)).strftime('%Y-%m-%d')), header=0)
        if (i == 0):
            df = temp
        else:
            df = pd.concat([df, temp], axis=0)

    return df

def drop_pairs_airlines(df):
    air_list = ['Air India', 'IndiGo', 'Spicejet','Vistara', 'Go Air','AirAsia']
    df = df.loc[df['airline'].isin(air_list)]
    return df

def hot_encode_flight_path(df):
    ''' 
    Return updated dataframe with one hot encoded flight paths

    Input: dataframe with 'departure_city' & 'arrival_city' as columns
    '''
    df['flight_path'] = df['departure_city'].str.cat(
        df['arrival_city'], sep="-")
    hot_encoding = pd.get_dummies(df['flight_path'])
    df.drop(columns=['departure_city', 'arrival_city'], inplace=True)
    df = pd.concat([df, hot_encoding], axis=1)
    return df


def hot_encode_airline(df):
    ''' 
    Return updated dataframe with one hot encoded airline

    Input: dataframe with 'airline' as column
    '''
    hot_encoding = pd.get_dummies(df['airline'])
    df = pd.concat([hot_encoding, df], axis=1)
    return df


def hot_encode_days(df):
    ''' 
    Return updated dataframe with one hot encoded days

    Input: dataframe with 'booking_day', 'departure_day' as column
    '''
    hot_encoding_bd = pd.get_dummies(df['booking_day'], prefix='bd_')
    hot_encoding_dd = pd.get_dummies(df['departure_day'], prefix='dd_')
    df = pd.concat([df, hot_encoding_bd, hot_encoding_dd], axis=1)
    return df


def hot_encode_clusters(df):
    '''
    Return updated dataframe with one hot encoded clusters based on departure_time[enum(morning, afternoon, evening, night)] and day of departure

    Input: dataframe with 'departure_time' & 'departure_day' as columns
    '''
    df['departure_day'] = df['departure_day'].astype(str)
    df['departure_time_day'] = df['departure_time'].str.cat(
        df['departure_day'], sep="-")
    hot_encoding = pd.get_dummies(df['departure_time_day'])
    df = pd.concat([df, hot_encoding], axis=1)
    df['departure_day'] = df['departure_day'].astype(int)
    return df


def drop_columns(df):
    ''' 
    Return updated dataframe with dropped columns

    Input: dataframe with 'flight_code', 'flight_path', 'departure_day', 'booking_day', 'airline', 'departure_date', 'booking_date' as columns
    '''
    df.drop(columns=['flight_code', 'flight_path',
                     'departure_day', 'booking_day', 'airline', 'departure_date', 'booking_date'], inplace=True)
    return df


# Order of execution
df = combine_input_files()
df = hot_encode_flight_path(df)
df = hot_encode_airline(df)
df = hot_encode_days(df)
df = hot_encode_clusters(df)
# Can plot graphs now

# Drop columns
df = drop_columns(df)
