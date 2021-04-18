import pandas as pd
from re import sub
from datetime import datetime
import preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import check_array
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import svm

import pandas_profiling


def clean_number_of_stops(stop):
    if (stop[0] == 'N'):
        return 0
    else:
        return int(stop[0])


def clean_price(price):

    return int(sub(r'[^\d.]', '', price))


def days_between(df):
    d1 = datetime.strptime(df['booking_date'], "%Y-%m-%d")
    d2 = datetime.strptime(df['departure_date'], "%Y-%m-%d")
    return abs((d2 - d1).days)


def add_days_left_departure(filename, departure_date):
    origindate = filename[filename.find("_")+1:filename.find(".")]
    return days_between(origindate, departure_date)


# print(add_days_left_departure("FlightsData_2020-09-29.csv","2020-10-5"))
# df = pd.read_csv("data_scraper\Data\FlightsData_2020-09-28.csv")

# print(df.apply(days_between, axis=1))

# print((data['flight_cost'].apply(clean_price)))
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = check_array(y_true, y_pred)

#     ## Note: does not handle mix 1d representation
#     #if _is_1d(y_true):
#     #    y_true, y_pred = _check_1d_array(y_true, y_pred)

#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


df = preprocess.preprocess(graphs=False)
# profile = pandas_profiling.ProfileReport(df)
# profile.to_file("your_report.html")
# print(df.info())
# df.plot(x='departure_date', y='days_to_depart')


def lol(df):
    sns.displot(data=df, x='flight_cost', hue='airline')


def depart_date_day_depart_price(df):
    plt.scatter(df['departure_date'], df['days_to_depart'],
                c=df['flight_cost'], s=10)
    plt.gray()
    plt.xlabel("Departure Date")
    plt.ylabel("Days To Departure")
    plt.tick_params(axis='x', labelsize=7)

    plt.show()


def graph_2(df):

    flights = df[['airline', 'days_to_depart', 'flight_cost']]

    names = ['Air India', 'IndiGo', 'Spicejet',
             'Vistara', 'Go Air']

    # prices_count=[[0 for i in range(5)] for j in range(5)]
    prices_count = {}

    for i in names:
        prices_count[i] = [0]*5

    for airline_name in names:
        prices_list = list(
            flights[flights['airline'] == airline_name]['flight_cost'])
        for i in prices_list:
            if i <= 3000:
                prices_count[airline_name][0] += 1
            elif i <= 4000 and i > 3000:
                prices_count[airline_name][1] += 1
            elif i <= 5000 and i > 4000:
                prices_count[airline_name][2] += 1
            elif i <= 6000 and i > 5000:
                prices_count[airline_name][3] += 1
            elif i > 6000:
                prices_count[airline_name][4] += 1

    print(prices_count)
    new_df = pd.DataFrame(prices_count)
    print(new_df)

    sns.displot(data=new_df, x=names, hue=new_df[names])

    plt.show()


def model(df):

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    df = df[df['flight_cost'] < 11000]
    # print(df.info())
# fit and transform the data
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    y = df['flight_cost']
    # X = df.loc[:, df.columns != 'flight_cost']
    X = df_norm.loc[:, df_norm.columns != 'flight_cost']
    # X = df[['flight_duration', 'number_of_stops', 'days_to_depart']]


    # X = df[['IndiGo', 'flight_duration', 'number_of_stops', 'days_to_depart',
    #          'Mumbai-Bengaluru', 'Mumbai-New Delhi', 'bd__3', 'bd__4', 'bd__5']]
    # X=df.drop('flight_cost')
    # no_of_rows=100000
    # X=X[:no_of_rows]
    # y=y[:no_of_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    # model = linear_model.LinearRegression()
    # model=linear_model.Ridge(alpha=0.5)
    # model=linear_model.Lasso(alpha=0.1,max_iter=5000)

    # for i in range(11,15):

    model = RandomForestRegressor(max_depth=15, random_state=1)
    # model = BaggingRegressor(base_estimator=RandomForestRegressor(max_depth=16),random_state=1)
    # model=svm.SVR(kernel='linear')

    # model = linear_model.BayesianRidge()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # print("Depth:",i)
    print("MSE ", mean_squared_error(y_test, pred))
    print("MAE", mean_absolute_error(y_test, pred))
    print("R2 Score ", r2_score(y_test, pred))
    print()
# plt.scatter(X_test, y_test, color='b')
# plt.plot(X_test, pred, color='k')

# plt.show()

# print(X_train.info())


def pca(df):

    y = df['flight_cost']
    X = df.loc[:, df.columns != 'flight_cost']

    df_std = StandardScaler().fit_transform(X)
    df_cov_matrix = np.cov(df_std.T)
    eig_vals, eig_vecs = np.linalg.eig(df_cov_matrix)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]
    total = sum(eig_vals)

    var_exp = [(i / total)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    # print("Variance captured by each component is \n", var_exp)

    print("Cumulative variance captured as we travel with each component \n", cum_var_exp)

    pca = PCA().fit(df_std)

    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('No of components')
    # plt.ylabel('Cumulative explained variance')
    # plt.show()

    pca = PCA(n_components=75)

    colums = []
    for i in range(1, 76):
        colums.append("PC"+str(i))
    pcs = pca.fit_transform(df_std)
    print(pcs.shape)
    df_new = pd.DataFrame(data=pcs, columns=colums)
    # df_new['price'] = y

    model = linear_model.LinearRegression()
    # model=linear_model.Ridge(alpha=0.5)
    # model=linear_model.Lasso(alpha=0.1)

    # model = linear_model.BayesianRidge()

    model.fit(df_new, y)
    print(model.score(df_new, y))
    pred = model.predict(df_new)
    print("MSE ", mean_absolute_error(y, pred))
    print("R2 Score ", r2_score(y, pred))


# pca(df)
model(df)
