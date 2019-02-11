import numpy as np
import pandas as pd
import statistics as stat
import datetime as d


class TimeDataFrame:

    def load(self):
        self.data_frame = pd.read_csv(self.file)
        self.keys = self.data_frame.keys().values

    def __init__(self, file, type='csv', time_key='Time'):
        self.file = file
        self.fileType = type
        self.time_key = time_key
        self.load()

    def reset(self, file, type='csv', time_key='Time'):
        self.file = file
        self.fileType = type
        self.time_key = time_key
        self.load()

    def get_time_key(self):
        return self.time_key

    def get_file(self):
        return self.file

    def fetch_keys(self):
        return self.keys

    def sample(self):
        print(self.data_frame.head())

    def fetch_series(self, key):
        time_col = self.data_frame[self.time_key]
        var_col = pd.to_numeric(self.data_frame[key], errors='coerce')
        return pd.Series(list(var_col), index=list(time_col))

    def add_day_column(self, row):
        return d.datetime.strptime(row['Time'], '%d.%m.%Y %H:%M').date().strftime("%A").lower()

    def get_day_of_week_series(self, key, day):
        series = self.fetch_series(key)
        day_df = series.to_frame().reset_index()
        day_df.columns = ['Time', 'Value']
        day_df['Day'] = day_df[['Time']].apply(self.add_day_column, axis=1)
        day_df = day_df[day_df.Day == day].reset_index()
        day_df.columns = ['OrigIndex', 'Time', 'Value', 'Day']
        return day_df['Value']

    def daily_sum(self, key):
        new_series = dict()
        u_series = self.fetch_series(key).fillna(0)
        keys = u_series.keys()
        for key in keys:
            date = key[:10]
            if date not in new_series:
                new_series[date] = u_series[key]
            else:
                new_series[date] += u_series[key]

        return pd.Series(new_series)

    def hourly_sum(self, key):
        new_series = dict()
        u_series = self.fetch_series(key).fillna(0)
        keys = u_series.keys()
        for key in keys:
            hour = key[:13]
            if hour not in new_series:
                new_series[hour] = u_series[key]
            else:
                new_series[hour] += u_series[key]

        return pd.Series(new_series)
