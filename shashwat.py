import datetime
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess


def get_time_delta(time):
    """
    Returns the time delta in minutes

    Input: Time of format `%H hrs %M mins ` or `%H hrs `
    """
    hrs_mins = re.findall(r'\d+', time)

    assert len(hrs_mins) > 0 and len(hrs_mins) < 3

    hours = int(hrs_mins[0])
    minutes = 0
    if len(hrs_mins) == 2:
        minutes = int(hrs_mins[1])

    time_delta = datetime.timedelta(hours=hours, minutes=minutes).seconds
    return time_delta/60


def get_part_of_day(hour):
    """
    Returns part of day for given hour

    Input:  hour (int)
    Output: Part of day (str) [morning (5 to 11), afternoon (12 to 17), evening (18 to 22), night (23 to 4)]
    """
    return (
        "morning" if 5 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 17
        else
        "evening" if 18 <= hour <= 22
        else
        "night"
    )


def classify_time(time):
    """
    Classify given time to part of day

    Input:  time in `HH:MM` format (str)
    Output: Part of day (str) [morning (5 to 11), afternoon (12 to 17), evening (18 to 22), night (23 to 4)]
    """
    hour = datetime.datetime.strptime(time, "%H:%M").time().hour
    return get_part_of_day(hour)


def drop_dates(df, inplace=True):
    """
    Drop departure dates and booking dates

    Input:  df (Dataframe) containing columns `departure_date` and `booking_date`
    Output: If inplace = True, return nothing
            If inplace = False, returns modified dataframe
    """
    return df.drop(["departure_date", "booking_date"], inplace=inplace)


# data = pd.read_csv("data_scraper\Data\FlightsData_2020-09-28.csv")
# print(data.columns)
# data['departure_time'] = data['departure_time'].apply(classify_time)
# data['flight_duration'] = data['flight_duration'].apply(get_time_delta)
# print(data.describe())

df = preprocess(graphs=not True)
print(df.info())

# print(df.info())
# df.plot(x='days_to_depart', y='flight_cost', kind='scatter')
# df.plot(x='departure_day', y='flight_cost', kind='scatter')
# df.plot(x='departure_time', y='flight_cost', kind='scatter')

# flights = df[['airline', 'days_to_depart', 'flight_cost']]
# print(df2.head())

# x1 = list(flights[flights['airline'] == 'Air India']['flight_cost'])
# x2 = list(flights[flights['airline'] == 'IndiGo']['flight_cost'])
# x3 = list(flights[flights['airline'] == 'Spicejet']['flight_cost'])
# x4 = list(flights[flights['airline'] == 'Vistara']['flight_cost'])
# x5 = list(flights[flights['airline'] == 'Go Air']['flight_cost'])

# # Assign colors for each airline and the names
# colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
# names = ['Air India', 'IndiGo', 'Spicejet',
#          'Vistara', 'Go Air']

# # Make the histogram using a list of lists
# # Normalize the flights and assign colors and names
# plt.hist([x1, x2, x3, x4, x5], bins=int(180/15),
#          color=colors, label=names, stacked=True)

# # Plot formatting
# plt.legend()

# plt.show()

# sns.displot(data=df, x='flight_cost', hue='airline')
# plt.show()
