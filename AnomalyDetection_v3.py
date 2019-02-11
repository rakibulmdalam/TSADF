import locale
import time
import datetime as d
import random
import math
from itertools import groupby
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib
font = {
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)
from matplotlib import pyplot as plt
from scipy.stats import mode as statmode
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from timedataframe import TimeDataFrame

class AD:
    '''
    tw = trend_window
    ma_w = moving average window
    '''
    def __init__(self, timeseries, freq, tw, ma_w, filename):

        self.freq = freq
        self.tw = tw
        self.ma_w = ma_w
        self.filename = filename
        self.timeseries = timeseries
        self.df = timeseries.to_frame().reset_index()  # create dataframe from given timeseries
        self.df.columns = ['time', 'value']   # add column names to dataframe

        self.preprocess()
        self._exec()

    def preprocess(self):
        self.df['diff'] = self.df['value'].diff()
        self.df['cs'] = 0
        self.df['wvs'] = 0
        self.df['qd'] = 0
        self.df['tps'] = 0
        self.df['q1'] = 0
        self.df['q3'] = 0
        self.df['diff_q1'] = 0
        self.df['diff_q3'] = 0
        self.df['value'].fillna(-1, inplace=True)
        #self.df.loc[self.df['value'] == -1,'cs'] = 1000  # marking NaN values as outliers

    def add_day_column(self, row):
        locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
        return d.datetime.strptime(row['time'], '%d.%m.%Y %H:%M').date().strftime("%A").lower()

    def _exec(self):
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        all_days = []
        for day in days:
            day = day.lower()
            self.df['day'] = self.df[['time']].apply(self.add_day_column, axis=1)
            day_df = self.df[self.df['day'] == day].reset_index()
            day_df.columns = ['oi', 'time', 'value', 'diff', 'cs', 'wvs', 'qd', 'tps', 'q1', 'q3', 'diff_q1', 'diff_q3', 'day']
            day_df = self._qd(day_df)
            day_df = self._diff_qd(day_df)
            all_days.append(day_df)
        
        self.new_df = pd.concat(all_days)
        #print(self.new_df)
        self.new_df = self.new_df.sort_values('oi').set_index('oi').reset_index(drop=True)
        self._adjust_diff_qd()
        self._tds()
        self._wvs()
        self._cs()
        self.write()

    def _tds(self):
        for l in range(len(self.new_df)):            
            rolling_series = self.new_df[l:l + self.tw]['value']
            tps = 0
            try:
                tps = self._set_tds(rolling_series)
            except Exception as e:
                pass
            self.new_df.at[(l + self.tw -1), 'tps'] = tps

    def _set_tds(self, series):
        
        X = series.diff().abs().dropna().values
        window = self.ma_w
        history = [X[i] for i in range(window)]
        test = [X[i] for i in range(window, len(X))]
        predictions = list()
        # walk forward over time steps in test
        for t in range(len(test)):
            length = len(history)
            yhat = np.mean([history[i] for i in range(length-window,length)])
            obs = test[t]
            predictions.append(yhat)
            history.append(obs)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        
        mae = mean_absolute_error(test[:-1], predictions[:-1])
        obs_error = abs(test[-1] - predictions[-1])
        deviation = obs_error - mae
        #wseries = pd.DataFrame({'s': series[3:], 'd': X[2:], 'a': test, 'p': predictions})
        #print(wseries)
        #print('mae {}'.format(mae))
        #print('obs_error {}'.format(obs_error))
        #print('deviation {}'.format(deviation)) 
        
        if deviation <= 0:
            deviation = 0
        
        return deviation

    def add_interval_column(self, row): 
        return str(row['time']).split(" ")[1]

    def _wvs(self):
        self.new_df['wvs'] = self.new_df.apply(self._set_wv, axis=1)
    
    def _set_wv(self, row):
        wv = 0
        if row['value'] < 0:
            wv = 1000
            
        return wv
        
    def _qd(self, day_df):
        day_df['interval'] = day_df[['time']].apply(self.add_interval_column, axis=1)
        intervals = day_df['interval'].unique().tolist()
        all_intervals = []
        for interval in intervals:
            interval_df = day_df[day_df['interval'] == interval]
            q1, q3 = np.percentile(interval_df[interval_df['value'] > -1]['value'].dropna().values, [25, 75])
            interval_df['q1'] = q1
            interval_df['q3'] = q3 
            interval_df['qd'] = interval_df.apply(self._set_qd, axis=1)
            all_intervals.append(interval_df)

        day_df = pd.concat(all_intervals)
        return day_df.sort_index()

    def _set_qd(self, row):
        qd = 0
        if row['value'] > row['q3']:
            qd = row['value'] - row['q3']
        elif row['value'] < 0:
            qd = 0
        elif row['value'] < row['q1']:
            qd = row['q1'] - row['value']
        return qd

    def _diff_qd(self, day_df):
        day_df['interval'] = day_df[['time']].apply(self.add_interval_column, axis=1)
        intervals = day_df['interval'].unique().tolist()
        all_intervals = []
        for interval in intervals:
            interval_df = day_df[day_df['interval'] == interval]
            q1, q3 = np.percentile(interval_df['diff'].dropna().values, [25, 75])
            interval_df['diff_q1'] = q1
            interval_df['diff_q3'] = q3 
            interval_df['diff_qd'] = interval_df.apply(self._set_diff_qd, axis=1)
            all_intervals.append(interval_df)

        day_df = pd.concat(all_intervals)
        return day_df.sort_index()

    def _adjust_diff_qd(self):
        first = self.new_df['diff_qd'][0]
        self.new_df['diff_qd'] = self.new_df['diff_qd'].diff()
        self.new_df['diff_qd'][0] = first

    def _set_diff_qd(self, row):
        qd = 0
        if row['diff'] > row['diff_q3']:
            qd = row['diff'] - row['diff_q3']
        elif row['diff'] < row['diff_q1']:
            qd = row['diff_q1'] - row['diff']
        return qd

    def _cs(self):
        self.new_df['cs'] = self.new_df.apply(self._set_cs, axis=1)

    def _set_cs(self, row):
        return row['qd'] + row['tps'] + row['wvs']

    def write(self):
        self.new_df.to_csv(self.filename)
