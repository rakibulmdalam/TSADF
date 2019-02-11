import locale
import time
import datetime as d
import random
import math
from itertools import groupby
from operator import itemgetter
from timedataframe import TimeDataFrame
from AnomalyDetection_v3 import AD
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import f


def plot_normalized_threshold(temp_df, threshold=0):
    _, splts = plt.subplots(3, sharex = True, sharey=True)
    index = list()
    range_high = list()
    range_low = list()
    threshold_high = list()
    threshold_low = list()
    diff_range_high = list()
    diff_range_low = list()
    diff_threshold_high = list()
    diff_threshold_low = list()
    cs = list()
    wvs = list()
    qd = list()
    tps = list()
    diff_qd = list()

    z2 = temp_df['normed_dqd'][2] / temp_df['diff_qd'][2]
    z1 = temp_df['normed_qd'][2] / temp_df['qd'][2]
    for i in temp_df.index.values:
        index.append(i)
        range_high.append(temp_df.loc[i]['q3'])
        range_low.append(temp_df.loc[i]['q1'])
        threshold_high.append(temp_df.loc[i]['q3'] + threshold / z1)
        threshold_low.append(temp_df.loc[i]['q1'] - threshold / z1)

        diff_range_high.append(temp_df.loc[i]['diff_q3'])
        diff_range_low.append(temp_df.loc[i]['diff_q1'])
        diff_threshold_high.append(temp_df.loc[i]['diff_q3'] + threshold / z2)
        diff_threshold_low.append(temp_df.loc[i]['diff_q1'] - threshold / z2)

        cs.append(temp_df.loc[i]['cs'])
        wvs.append(temp_df.loc[i]['wvs'])
        qd.append(temp_df.loc[i]['qd'])
        diff_qd.append(temp_df.loc[i]['diff_qd'])
        tps.append(temp_df.loc[i]['tps'])
    
    splts[0].fill_between(index, threshold_high, range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(threshold))
    splts[0].fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    splts[0].fill_between(index, range_low, threshold_low, color='lightblue', alpha=0.4)
    temp_df['value'].plot(ax=splts[0], color='mediumslateblue')
    s = temp_df['value'][temp_df['normed_qd'] > threshold]
    splts[0].scatter(y=s.values, x=s.index.values, color='red', label='Anomalies')
    splts[0].set_title('Anomalies against QD for threshold {}'.format(threshold))
    splts[0].legend()


    splts[1].fill_between(index, diff_threshold_high, diff_range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(threshold))
    splts[1].fill_between(index, diff_range_high, diff_range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    splts[1].fill_between(index, diff_range_low, diff_threshold_low, color='lightblue', alpha=0.4)
    temp_df['diff'].plot(ax=splts[1], color='mediumslateblue')
    s_diff = temp_df['diff'][(temp_df['normed_dqd'] > threshold)]
    splts[1].scatter(y=s_diff.values, x=s_diff.index.values, color='red', label='Anomalies')
    splts[1].set_title('Anomalies against Difference QD for threshold {}'.format(threshold))
    splts[1].legend()

    splts[2].fill_between(index, threshold_high, range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(threshold))
    splts[2].fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    splts[2].fill_between(index, range_low, threshold_low, color='lightblue', alpha=0.4)
    temp_df['value'].plot(ax=splts[2], color='mediumslateblue')
    s_combined = temp_df['value'][temp_df['wvs'] > threshold]
    splts[2].scatter(y=s_combined.values, x=s_combined.index.values, color='red', label='Anomalies')
    splts[2].set_title('Anomalies against WVS for threshold {}'.format(threshold))
    splts[2].legend()

    #plt.xticks(index, time_ticks, rotation=90)
    plt.show()



def plot(temp_df, threshold=0):
    _, splts = plt.subplots(3, sharex = True, sharey=True)
    index = list()
    range_high = list()
    range_low = list()
    threshold_high = list()
    threshold_low = list()
    diff_range_high = list()
    diff_range_low = list()
    diff_threshold_high = list()
    diff_threshold_low = list()
    cs = list()
    wvs = list()
    qd = list()
    tps = list()
    diff_qd = list()
    for i in temp_df.index.values:
        index.append(i)
        range_high.append(temp_df.loc[i]['q3'])
        range_low.append(temp_df.loc[i]['q1'])
        threshold_high.append(temp_df.loc[i]['q3'] + threshold)
        threshold_low.append(temp_df.loc[i]['q1'] - threshold)

        diff_range_high.append(temp_df.loc[i]['diff_q3'])
        diff_range_low.append(temp_df.loc[i]['diff_q1'])
        diff_threshold_high.append(temp_df.loc[i]['diff_q3'] + threshold)
        diff_threshold_low.append(temp_df.loc[i]['diff_q1'] - threshold)

        cs.append(temp_df.loc[i]['cs'])
        wvs.append(temp_df.loc[i]['wvs'])
        qd.append(temp_df.loc[i]['qd'])
        diff_qd.append(temp_df.loc[i]['diff_qd'])
        tps.append(temp_df.loc[i]['tps'])
    
    splts[0].fill_between(index, threshold_high, range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(threshold))
    splts[0].fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    splts[0].fill_between(index, range_low, threshold_low, color='lightblue', alpha=0.4)
    temp_df['value'].plot(ax=splts[0], color='mediumslateblue')
    s = temp_df['value'][temp_df['qd'] > threshold]
    splts[0].scatter(y=s.values, x=s.index.values, color='red', label='Anomalies')
    splts[0].set_title('Anomalies against QD for threshold {}'.format(threshold))
    splts[0].legend()


    splts[1].fill_between(index, diff_threshold_high, diff_range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(threshold))
    splts[1].fill_between(index, diff_range_high, diff_range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    splts[1].fill_between(index, diff_range_low, diff_threshold_low, color='lightblue', alpha=0.4)
    temp_df['diff'].plot(ax=splts[1], color='mediumslateblue')
    s_diff = temp_df['diff'][(temp_df['diff_qd'] > threshold)]
    splts[1].scatter(y=s_diff.values, x=s_diff.index.values, color='red', label='Anomalies')
    splts[1].set_title('Anomalies against Difference QD for threshold {}'.format(threshold))
    splts[1].legend()

    splts[2].fill_between(index, threshold_high, range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(threshold))
    splts[2].fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    splts[2].fill_between(index, range_low, threshold_low, color='lightblue', alpha=0.4)
    temp_df['value'].plot(ax=splts[2], color='mediumslateblue')
    s_combined = temp_df['value'][temp_df['wvs'] > threshold]
    splts[2].scatter(y=s_combined.values, x=s_combined.index.values, color='red', label='Anomalies')
    splts[2].set_title('Anomalies against WVS for threshold {}'.format(threshold))
    splts[2].legend()

    #plt.xticks(index, time_ticks, rotation=90)
    plt.show()



def final_plot(temp_df, qd_t, dqd_t):
    _, splts = plt.subplots(1, sharex = True, sharey=True)
    index = list()
    range_high = list()
    range_low = list()
    threshold_high = list()
    threshold_low = list()
    diff_range_high = list()
    diff_range_low = list()
    diff_threshold_high = list()
    diff_threshold_low = list()
    qd = list()
    diff_qd = list()
    for i in temp_df.index.values:
        index.append(i)
        range_high.append(temp_df.loc[i]['q3'])
        range_low.append(temp_df.loc[i]['q1'])
        threshold_high.append(temp_df.loc[i]['q3'] + qd_t)
        threshold_low.append(temp_df.loc[i]['q1'] - qd_t)

        diff_range_high.append(temp_df.loc[i]['diff_q3'])
        diff_range_low.append(temp_df.loc[i]['diff_q1'])
        diff_threshold_high.append(temp_df.loc[i]['diff_q3'] + dqd_t)
        diff_threshold_low.append(temp_df.loc[i]['diff_q1'] - dqd_t)


        qd.append(temp_df.loc[i]['qd'])
        diff_qd.append(temp_df.loc[i]['diff_qd'])
    
    #splts.fill_between(index, threshold_high, range_high, color='lightblue', alpha=0.3, label='{} quartile distance'.format(qd_t))
    #splts.fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
    #splts.fill_between(index, range_low, threshold_low, color='lightblue', alpha=0.4)
    temp_df['value'].plot(ax=splts, color='mediumslateblue')
    s = temp_df[temp_df['qd'] > qd_t]['value']
    splts.scatter(y=s.values, x=s.index.values, color='red', marker='o', label='Anomalies based on point-distance')

    s2 = temp_df[temp_df['diff_qd'] > dqd_t]['value']
    splts.scatter(y=s2.values, x=s2.index.values, color='orange', marker='x', label='Anomalies based on difference-distance')

    s3 = temp_df[temp_df['wvs'] == 1000]['value']
    splts.scatter(y=s3.values, x=s3.index.values, color='black', marker='s', label='Non-contextual anomalies')

    splts.set_title('Anomalies against threshold pair ({}, {})'.format(qd_t, dqd_t))
    splts.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=_.transFigure, ncol=3)
    plt.show()

    _, splts = plt.subplots(1, sharex = True, sharey=True)
    splts.fill_between(index, threshold_high, range_high, color='lightblue', alpha=0.3, label='{} point distance border'.format(qd_t))
    splts.fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior')
    splts.fill_between(index, range_low, threshold_low, color='lightblue', alpha=0.4)
    temp_df['value'].plot(ax=splts, color='mediumslateblue')
    s_diff = temp_df['value'][(temp_df['qd'] > qd_t)]
    splts.scatter(y=s_diff.values, x=s_diff.index.values, color='orange', marker='x', label='Anomalies')
    splts.set_title('Anomalies against point distance threshold {}'.format(qd_t))
    splts.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=_.transFigure, ncol=3)

    _, splts = plt.subplots(1, sharex = True, sharey=True)
    splts.fill_between(index, diff_threshold_high, diff_range_high, color='lightblue', alpha=0.3, label='{} difference distance border'.format(dqd_t))
    splts.fill_between(index, diff_range_high, diff_range_low, color='lightblue', alpha=0.8, label='Normal Behavior')
    splts.fill_between(index, diff_range_low, diff_threshold_low, color='lightblue', alpha=0.4)
    temp_df['diff'].plot(ax=splts, color='mediumslateblue')
    s_diff = temp_df['diff'][(temp_df['diff_qd'] > dqd_t)]
    splts.scatter(y=s_diff.values, x=s_diff.index.values, color='orange', marker='x', label='Anomalies')
    splts.set_title('Anomalies against difference distance threshold {}'.format(dqd_t))
    splts.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=_.transFigure, ncol=3)

    #plt.xticks(index, time_ticks, rotation=90)
    plt.show()



# final_plot(raw_df[0:2000], 204, 182)
# final_plot(raw_df[0:2000], 47, 29)

keys = ['MA33_TEU_200_202_204','MA33_12001_R1', 'MA33_5028_R2', 'MA33_2055_R1']
files = ['raw_data_files/MA33_2011_1_15.csv','raw_data_files/MA33_2011_1_15.csv', 'raw_data_files/MA33_2011_2_15.csv', 'raw_data_files/MA33_2011_2_15.csv']

ld = 'LD1'

a = 3
TDF = TimeDataFrame(files[a])
key_series = TDF.fetch_series(keys[a])
key_series.to_csv('MA33_2055_R1.csv')
# phase 1
# Missing and out of range values
# LD1_missing = 1505
# LD2_missing = 2079
# LD3_missing = 3226
# LD4_missing = 4368
out_of_range_count = len(key_series) - key_series.describe()['count']



######################################
######################################
######################################
######################################
#phase 2


def compare(X, Y, name_x, name_y, ticks, ld):

    c_collector = []
    unexpected = 0
    insignificant = 0
    C = 0
    for d in range(len(X)): # 96
        D = X[d] 
        var_D = np.var(D, ddof=1)

        # 3. create blocks and collect all D_i
        c = 0
        for i in range(len(Y)): # 672
            D_i = Y[i][d]
            # determine variance
            var_D_i = np.var(D_i, ddof=1)
            # check variance
            if var_D_i < var_D:
                # check significance of the difference
                deg_D = len(D) - 1
                deg_D_i = len(D_i) - 1
                alpha = 0.05                        
                critical_val_low = f.ppf(q=alpha/2, dfn=deg_D, dfd=deg_D_i)
                critical_val_high = f.ppf(q=1 - alpha/2, dfn=deg_D, dfd=deg_D_i)
                
                fstat_sample = var_D / var_D_i
                
                if (fstat_sample < critical_val_low) | (fstat_sample > critical_val_high):
                    c = c + 1

                else:
                    unexpected = unexpected + 1
                    insignificant = insignificant + 1
            else:
                unexpected = unexpected + 1

        #print(c)
        c_collector.append(c)
        if c > int(len(Y)/2):
            C = C + 1
    
    #print(C)
    winner = ''
    if C > int(len(X)/2):
        print('Go with {}'.format(name_y))
        winner = name_y
    else:
        print('Go with {}'.format(name_x))
        winner = name_x

    print('unexpected variance {}'.format(unexpected))



    # fig, splots = plt.subplots(2, figsize=(10,8))
    # splots[0].bar(range(len(c_collector)), c_collector)
    # splots[0].set_title('Votes for {}'.format(name_y))
    # splots[0].set_xlabel('Freq of {}'.format(name_x))
    # splots[0].set_ylabel('Votes')
    # splots[0].grid(True)

    #fig, splots = plt.subplots(1, figsize=(10,8))
    plt.bar(range(len(c_collector)), c_collector)
    plt.title('{}: Votes for {} seasonality'.format(ld, name_y))
    plt.xticks([i for i in range(96)], ticks, rotation=90)
    plt.xlabel('{} data-points'.format(name_x))
    plt.ylabel('Votes')
    #plt.grid(True)
    plt.tight_layout()
    plt.show()

    labels = ['{} var not smaller than {} \n'.format(name_y, name_x), 'insignificant difference', '{} var is significantly \nsmaller than {}'.format(name_y, name_x)]
    insignificant_var_perc = insignificant / (len(Y) * len(Y[0])) * 100
    unexpected_var_perc = unexpected / (len(Y) * len(Y[0])) * 100
    expected_var_perc = 100 - unexpected_var_perc
    sizes = [unexpected_var_perc - insignificant_var_perc, insignificant_var_perc, expected_var_perc]

    explode = (0, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'unexpected')

    
    patches, texts, _ = plt.pie(sizes, explode=explode, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(patches, labels, loc="best")
    plt.title('{}'.format(ld))
    plt.show()

    return winner, sizes, c_collector, [len(Y), len(Y[0])]



def add_day_column(row):
    locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
    return d.datetime.strptime(row['time'], '%d.%m.%Y %H:%M').date().strftime("%A").lower()

def add_interval_column(row): 
    return str(row['time']).split(" ")[1]

def add_hour_column(row): 
    return int(str(row['interval']).split(":")[0])

def add_minute_column(row): 
    return str(row['interval']).split(":")[1]

def add_date_column(row): 
    return str(row['time']).split(" ")[0]

def add_day_of_month_column(row):
    return int(str(row['date']).split(".")[0])


def add_month_index_column(row):
    return str(row['date']).split(".")[1]



############################# P E R F E C T ##################################################

ld = 'LD4'

a = 3
TDF = TimeDataFrame(files[a])
key_series = TDF.fetch_series(keys[a])


# prepare X for daily vs weekly comparison
df = key_series.to_frame().reset_index()
df.columns = ['time', 'value']
df['day'] = df[['time']].apply(add_day_column, axis=1)
df['interval'] = df[['time']].apply(add_interval_column, axis=1)
df['hod'] = df[['interval']].apply(add_hour_column, axis=1)
df['moh'] = df[['interval']].apply(add_minute_column, axis=1)
df['date'] = df[['time']].apply(add_date_column, axis=1)
df['dom'] = df[['date']].apply(add_day_of_month_column, axis=1)
df['wom'] = df['dom'] / 7
df['wom'] = np.ceil(df['wom'])
df['wom'] = df['wom'] - 1


day_counter = 0
for date in df['date'].unique().tolist():
    #copy_df.loc[copy_df['date'] == date, 'add_group_index'] = day_counter
    df.loc[df['date'] == date, 'doy'] = day_counter
    day_counter = day_counter + 1


df['month_index'] = df[['date']].apply(add_month_index_column, axis=1)
df['diff'] = df['value'].diff()
intervals = df['interval'].unique().tolist()
doms = df['dom'].unique().tolist()

copy_df = df
copy_df = copy_df.reset_index(drop=True).reset_index()
X = []

for i in copy_df['interval'].unique().tolist():
    s = copy_df[copy_df['interval'] == i]['value']
    # print(s)
    X.append(s.dropna().values.tolist())



#Y = weekly blocks
copy_df = df
copy_df = copy_df.reset_index(drop=True).reset_index()

Y = []

for ind in copy_df['day'].unique().tolist():
    temp = copy_df[copy_df['day'] == ind] # ind = 'monday', 'tuesday', 'wednesday', 'thursday'
    all_s = [] 
    for i in temp['interval'].unique().tolist(): # 0-> 00:00, 1 -> 00:15
        s = temp[temp['interval'] == i]['value']
        # print(s)
        all_s.append(s.dropna().values.tolist())
    
    Y.append(all_s)

ticks = df['interval'].unique().tolist()
for i in range(96):
    if i % 16 != 0:
        ticks[i] = ''


winner2, sizes2, c_collector2, Y_dim2 = compare(X,Y,'DAILY', 'WEEKLY', ticks, ld)

# unexpected variance 277 LD1
# unexpected variance 236 LD2
# unexpected variance 238 LD3
# unexpected variance 234 LD4

# putting together all pie charts data

from matplotlib import rc
 
# Data
r = [0,1,2,3]
 
# From raw value to percentage
greenBars = [58.8, 64.9, 64.6, 65.2]
orangeBars = [25.4, 28, 30.1, 25.4]
blueBars = [15.8, 7.1, 5.4, 9.4]
 
# plot
barWidth = 0.85
names = ('LD1','LD2','LD3','LD4')
# Create green Bars
plt.bar(r, greenBars, width=barWidth, label='Weekly VAR significantly smaller than daily')
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, width=barWidth, label='Weekly VAR smaller but not significant')
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], width=barWidth, label='Weekly VAR not smaller than daily')
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Loop Detectors")
plt.legend()
plt.grid(True)
# Show graphic
plt.show()

######################################
######################################
######################################
######################################
######################################

# phase 3 & 4
#abs_series = key_series.diff().abs()
ad = AD(key_series, 96, 12, 4, '{}_scores.csv'.format(ld))


#final_plot(raw_df[0:2000], 204, 182)
#final_plot(raw_df[0:2000], 47, 29)
#LD1_LD2_Score_Dist.png

ld = 'LD1'
an1_df = pd.read_csv('{}_scores.csv'.format(ld))
an1_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']

ld = 'LD2'
an2_df = pd.read_csv('{}_scores.csv'.format(ld))
an2_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']

ld = 'LD3'
an1_df = pd.read_csv('{}_scores.csv'.format(ld))
an1_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']

ld = 'LD4'
an2_df = pd.read_csv('{}_scores.csv'.format(ld))
an2_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']

# normalization process
# def normalize_qd(row, min_x, max_x):
#     z = (row - min_x) / (max_x - min_x) * 100
#     return z


#an1_df['normed_qd'] = an_df['qd'].apply(normalize_qd, args=(an_df['qd'].describe()['min'], an_df['qd'].describe()['max'], ))

#an1_df['normed_dqd'] = an_df['diff_qd'].apply(normalize_qd, args=(an_df['diff_qd'].describe()['min'], an_df['diff_qd'].describe()['max'], ))


# for i in range(1):
#     start = i*5000
#     end = i*5000 + 5000
#     #plot(an_df[start:end], 10)
#     plot_normalized_threshold(an_df[start:end], 20)



# # histograms
# plt.hist(an_df[an_df['qd']>0]['qd'], bins=12)
# #plt.xlim([-50,600])
# plt.grid(True)
# plt.xlabel('point-distance score')
# plt.ylabel('number of potential anomalies')
# plt.show()

# plt.hist(an_df[an_df['diff_qd']>0]['diff_qd'], bins=12)
# #plt.xlim([-50,600])
# plt.grid(True)
# plt.xlabel('difference-distance score')
# plt.ylabel('number of potential anomalies')
# plt.show()


fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(9, 4), sharey=True)

# generate some random test data
all_data = [an1_df[an1_df['qd']>0]['qd'].values.tolist(), an1_df[an1_df['diff_qd']>0]['diff_qd'].values.tolist()]
all_data2 = [an2_df[an2_df['qd']>0]['qd'].values.tolist(), an2_df[an2_df['diff_qd']>0]['diff_qd'].values.tolist()]

# plot violin plot
axes[0].violinplot(all_data,
                   showmeans=False,
                   showmedians=True)
axes[0].set_title('LD1: Score Distribution')


axes[0].yaxis.grid(True)
axes[0].set_xticks([y+1 for y in range(len(all_data))])
axes[0].set_xlabel('scores')
axes[0].set_ylabel('number of potential anomalies')

axes[1].violinplot(all_data2,
                   showmeans=False,
                   showmedians=True)
axes[1].set_title('LD2: Score Distribution')


axes[1].yaxis.grid(True)
axes[1].set_xticks([y+1 for y in range(len(all_data))])
axes[1].set_xlabel('scores')
#axes[1].set_ylabel('number of potential anomalies')

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['DS', 'DDS'])

plt.ylim(-10, 1200)
plt.show()


######################################
######################################
######################################
######################################
# phase 5 & 6

# Interactive Threshold Selection

def detailed_plot(raw_df, col, val_col, an_index, data_points=300, threshold=0, anomaly_count=0):

    
    f, splts = plt.subplots(len(an_index), 2, sharey=True)
    for j in range(len(an_index)):

        # show data points around the selected points
        index = list()
        range_high = list()
        range_low = list()
        threshold_high = list()
        threshold_low = list()
        cs = list()
        wvs = list()
        qd = list()
        dqd = list()
        tps = list()

        block_start = 0
        block_end = len(df)
        if int(an_index[j] - data_points / 2) >= 0:
            block_start = int(an_index[j] - data_points / 2)
        
        if int(an_index[j] + data_points / 2) <= len(df):
            block_end = int(an_index[j] + data_points / 2)

        print('{}, {}, {}'.format(block_start, block_end, an_index[j]))

        try:
            
            temp_df = raw_df[block_start:block_end]
            temp_df = temp_df.reset_index(drop=True)

            for i in temp_df.index.values:
                index.append(i)
                if val_col == 'diff':
                    range_high.append(temp_df.loc[i]['diff_q3'])
                    range_low.append(temp_df.loc[i]['diff_q1'])
                    threshold_high.append(temp_df.loc[i]['diff_q3'] + threshold)
                    threshold_low.append(temp_df.loc[i]['diff_q1'] - threshold)
                else: 
                    range_high.append(temp_df.loc[i]['q3'])
                    range_low.append(temp_df.loc[i]['q1'])
                    threshold_high.append(temp_df.loc[i]['q3'] + threshold)
                    threshold_low.append(temp_df.loc[i]['q1'] - threshold)

                cs.append(temp_df.loc[i]['cs'])
                wvs.append(temp_df.loc[i]['wvs'])
                qd.append(temp_df.loc[i]['qd'])
                dqd.append(temp_df.loc[i]['diff_qd'])
                tps.append(temp_df.loc[i]['tps'])
            
            splts[j][0].fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
            temp_df[val_col].plot(ax=splts[j][0], color='green')
            s = temp_df[temp_df['oi'] == an_index[j]][val_col]
            splts[j][0].scatter(y=s.values, x=s.index.values, color='red', label='Anomalies')
            #splts[j][0].set_title('Anomalies against threshold {}'.format(threshold))
            #splts[j][0].legend()
            splts[j][0].grid(True)

            #ticks = [ x for x in xticks if x.split()[1] == '00:00']
            temp_df['show_ticks'] = False
            temp_df['show_ticks'] = temp_df['oi'].apply(lambda x: (x % 6 == 0))
            #print(temp_df[temp_df['oi'] == an_index[j]]['show_ticks'])
            ticks = temp_df[temp_df['show_ticks']]['time']
            splts[j][0].set_xticklabels(ticks.values.tolist(),rotation=5)

        except:
            print('out of boundary expception occurred in searching...')

        
        # show the cluster this data point belongs to
        day = raw_df[raw_df['oi'] == an_index[j]]['day'].values.tolist()[0]
        interval = raw_df[raw_df['oi'] == an_index[j]]['interval'].values.tolist()[0]
        cluster_df = raw_df[(raw_df['day'] == day) & (raw_df['interval'] == interval)]
        cluster_df = cluster_df.reset_index(drop=True)
        # show all data points in a time-series manner
        splts[j][1].scatter(y = cluster_df[val_col].values, x = cluster_df.index.values, color='mediumslateblue')
        q1, q3 = np.percentile(cluster_df[cluster_df['value'] > -1][val_col].dropna().values, [25, 75])
        splts[j][1].fill_between([v for v in range(len(cluster_df))], [q3 for v in range(len(cluster_df))], [q1 for v in range(len(cluster_df))], color="lightblue", alpha=0.8, label='Normal Behavior')
        # mark the data point
        anomaly_series = cluster_df[cluster_df['oi'] == an_index[j]][val_col]
        splts[j][1].scatter(y=anomaly_series.values, x = anomaly_series.index.values, color='red', label='Anomaly')
        splts[j][1].grid(True)
        
        
    plt.subplots_adjust(top=0.94,bottom=0.075,left=0.04,right=0.97,hspace=0.285,wspace=0.095)
    f.suptitle('Current threshold {}, # of anomalies detected {}'.format(threshold, anomaly_count))
    plt.show()



def determine_threshold(raw_df, score_type):
    value_col = 'value'
    if score_type == 'qds':
        col = 'qd'
        
    elif score_type == 'tds':
        col = 'tps'

    elif score_type == 'diff_qds':
        col = 'diff_qd'
        value_col = 'diff'
    
    threshold_high = raw_df.describe()[col]['max']
    threshold_low = 0
    valid_threshold = 0

    intermediate_results = []
    while (threshold_high > (threshold_low+1)):
        threshold_mean = math.ceil((threshold_high + threshold_low) / 2)

        an_list = raw_df[raw_df[col] > threshold_mean]
        an_list = an_list.sort_values(col)
        anomaly_count = len(an_list)
        print('threshold_high {}, threshold_mean {}, threshold_low{}'.format(threshold_high, threshold_mean, threshold_low))
        closest_five = an_list['oi'][:5].values.tolist()
        print(closest_five)

        data_points = 100
        detailed_plot(raw_df, col, value_col, closest_five, data_points, threshold_mean, anomaly_count)
        choice = input('Do you think most of the red points are outliers? (y / n): ')
        
        if choice == 'y':
            intermediate_results.append({'threshold': threshold_mean, 'anomalies': len(an_list)})
            threshold_high = threshold_mean
            #threshold_mean = (threshold_mean + threshold_low) / 2
            valid_threshold = threshold_mean
        elif choice == 'q':
            break
        else:
            threshold_low = threshold_mean
            #threshold_mean = (threshold_mean + threshold_high) / 2
            

    return valid_threshold, intermediate_results



an_df = pd.read_csv('{}_scores.csv'.format(ld))
an_df = an_df[0:35040]
an_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']


ds_threshold, r1 = determine_threshold(an_df, 'qds')
dds_threshold, r2 = determine_threshold(an_df, 'diff_qds')


############################# counting anomalies ###################


ld = 'LD1'
an_df = pd.read_csv('{}_scores.csv'.format(ld))
an_df = an_df[0:35040]
an_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']


ds_threshold, r1 = determine_threshold(an_df, 'qds')
dds_threshold, r2 = determine_threshold(an_df, 'diff_qds')


# count anomalies


 
point_an_df = an_df[an_df['qd'] > 42]
diff_an_df = an_df[an_df['diff_qd'] > 36]

only_point_an_df = point_an_df[point_an_df['diff_qd'] <= 36]
only_diff_an_df = diff_an_df[diff_an_df['qd'] <= 42]

print('ds_anomaly = {}, dds_anomaly = {}, common_anomaly = {}'.format(len(only_point_an_df), len(only_diff_an_df), len(point_an_df) - len(only_point_an_df) ) )


# additional calculation for mz


def set_mz_qd(r, med, mad):
    return 0.6745 * (r['qd'] - med) / mad

def set_mz_dqd(r, med, mad):
    return 0.6745 * (r['diff_qd'] - med) / mad


ld = 'LD3'
an_df = pd.read_csv('{}_scores.csv'.format(ld))
an_df = an_df[0:35040]
an_df.columns = ['oi','time','value','diff','cs','wvs','qd','tps','q1','q3','diff_q1','diff_q3','day','interval','diff_qd']


an_df['qd_mz'] = 0
series = an_df[an_df['qd'] > 0]['qd'].dropna().values.tolist()
med = np.median(series)
mad = np.median([np.abs(y - med) for y in series])
an_df['qd_mz'] = an_df[['qd']].apply(set_mz_qd, args=(med,mad,), axis=1)

an_df['dqd_mz'] = 0
series = an_df[an_df['diff_qd'] > 0]['diff_qd'].dropna().values.tolist()
med = np.median(series)
mad = np.median([np.abs(y - med) for y in series])
an_df['dqd_mz'] = an_df[['diff_qd']].apply(set_mz_dqd, args=(med,mad,), axis=1)

 

point_an_df = an_df[an_df['qd_mz'] > 3.5]
diff_an_df = an_df[an_df['dqd_mz'] > 3.5]

only_point_an_df = point_an_df[point_an_df['dqd_mz'] <= 3.5]
only_diff_an_df = diff_an_df[diff_an_df['qd_mz'] <= 3.5]

print('ds_anomaly = {}, dds_anomaly = {}, common_anomaly = {}'.format(len(only_point_an_df), len(only_diff_an_df), len(point_an_df) - len(only_point_an_df) ) )

# ld = 'LD1'
# {'ds_counts' : 232, 'dds_counts' : 56, 'common_counts' : 168, 'range_counts' : 1505}, 1961, 6%
# {'ds_counts' : 2372, 'dds_counts' : 1809, 'common_counts' : 1000, 'range_counts' : 1505}, 6686, 20%
# {'ds_counts' : 1075, 'dds_counts' : 420, 'common_counts' : 398, 'range_counts' : 1505}, 3398, 10%

# ld = 'LD2'
# high threshold
# [{'threshold': 260, 'anomalies': 123}, {'threshold': 130, 'anomalies': 495}, {'threshold': 98, 'anomalies': 843}]
# [{'threshold': 252, 'anomalies': 18}, {'threshold': 126, 'anomalies': 80}, {'threshold': 95, 'anomalies': 196}, {'threshold': 87, 'anomalies': 265}, {'threshold': 85, 'anomalies': 288}]
# exclusive: ds_anomaly = 745, dds_anomaly = 190, common_anomaly = 98
# 3112, 9%


# low threshold
# [{'threshold': 260, 'anomalies': 123}, {'threshold': 130, 'anomalies': 495}, {'threshold': 98, 'anomalies': 843}, {'threshold': 82, 'anomalies': 1188}, {'threshold': 74, 'anomalies': 1441}, {'threshold': 70, 'anomalies': 1579}]
# [{'threshold': 252, 'anomalies': 18}, {'threshold': 126, 'anomalies': 80}, {'threshold': 63, 'anomalies': 676}, {'threshold': 48, 'anomalies': 1300}, {'threshold': 40, 'anomalies': 1902}, {'threshold': 36, 'anomalies': 2310}, {'threshold': 35, 'anomalies': 2416}]
# exclusive: ds_anomaly = 1129, dds_anomaly = 1966, common_anomaly = 450
# 5624, 16%

# mz threshold
# ex: ds_anomaly = 973, dds_anomaly = 283, common_anomaly = 147
# 3482, 10%

# ld = 'LD3'
# high threshold
# [{'threshold': 150, 'anomalies': 152}, {'threshold': 113, 'anomalies': 254}, {'threshold': 104, 'anomalies': 284}, {'threshold': 99, 'anomalies': 296}, {'threshold': 97, 'anomalies': 311}]
# [{'threshold': 148, 'anomalies': 18}, {'threshold': 74, 'anomalies': 70}, {'threshold': 70,'anomalies': 82}]
# exclusive: ds_anomaly = 288, dds_anomaly = 59, common_anomaly = 23, out-range = 3226
# 11% 

# Low threshold
# [{'threshold': 150, 'anomalies': 152}, {'threshold': 75, 'anomalies': 472}, {'threshold': 57, 'anomalies': 744}, {'threshold': 48, 'anomalies': 1021}, {'threshold': 43, 'anomalies': 1274}, {'threshold': 41, 'anomalies': 1376}]
# [{'threshold': 148, 'anomalies': 18}, {'threshold': 74, 'anomalies': 70}, {'threshold': 37,'anomalies': 731}, {'threshold': 28, 'anomalies': 1399}, {'threshold': 26, 'anomalies': 1609}, {'threshold': 25, 'anomalies': 1758}]
# exclusive: ds_anomaly = 1049, dds_anomaly = 1431, common_anomaly = 327, out-range = 3226
# 18%

# mz threshold
#  ex: ds_anomaly = 911, dds_anomaly = 323, common_anomaly = 110, out-range = 3226
# 13%


# ld = 'LD4'
# high threshold
# [{'threshold': 234, 'anomalies': 150}, {'threshold': 176, 'anomalies': 357}, {'threshold': 147, 'anomalies': 504}, {'threshold': 132, 'anomalies': 606}, {'threshold': 125, 'anomalies': 647}, {'threshold': 123, 'anomalies': 659}]
# [{'threshold': 239, 'anomalies': 76}, {'threshold': 120, 'anomalies': 313}, {'threshold': 105, 'anomalies': 372}, {'threshold': 98, 'anomalies': 389}, {'threshold': 94, 'anomalies': 401}, {'threshold': 92, 'anomalies': 406}, {'threshold': 91, 'anomalies': 407}]
# exclusive: ds_anomaly = 443, dds_anomaly = 191, common_anomaly = 216, out-range = 4368
# 15%

# Low threshold
# [{'threshold': 234, 'anomalies': 150}, {'threshold': 117, 'anomalies': 709}, {'threshold': 59, 'anomalies': 1515}, {'threshold': 45, 'anomalies': 2024}, {'threshold': 42, 'anomalies':2183}]
# [{'threshold': 239, 'anomalies': 76}, {'threshold': 120, 'anomalies': 313}, {'threshold': 60, 'anomalies': 589}, {'threshold': 45, 'anomalies': 752}, {'threshold': 38, 'anomalies': 916}, {'threshold': 36, 'anomalies': 975}]
# exclusive: ds_anomaly = 1712, dds_anomaly = 504, common_anomaly = 471, out-range = 4368
# 20%

# mz threshold
# ds_anomaly = 1324, dds_anomaly = 464, common_anomaly = 412, out-range = 4368
# 19%

# plot counts
LD1 = [
    {'ds_counts' : 232, 'dds_counts' : 56, 'common_counts' : 168, 'range_counts' : 1505},
    {'ds_counts' : 2372, 'dds_counts' : 1809, 'common_counts' : 1000, 'range_counts' : 1505},
    {'ds_counts' : 1075, 'dds_counts' : 420, 'common_counts' : 398, 'range_counts' : 1505}
]

LD2 = [
    {'ds_counts' : 745, 'dds_counts' : 190, 'common_counts' : 98, 'range_counts' : 2079},
    {'ds_counts' : 1129, 'dds_counts' : 1966, 'common_counts' : 450, 'range_counts' : 2079},
    {'ds_counts' : 973, 'dds_counts' : 283, 'common_counts' : 147, 'range_counts' : 2079}
]

LD3 = [
    {'ds_counts' : 288, 'dds_counts' : 59, 'common_counts' : 23, 'range_counts' : 3226},
    {'ds_counts' : 1049, 'dds_counts' : 1431, 'common_counts' : 327, 'range_counts' : 3226},
    {'ds_counts' : 911, 'dds_counts' : 323, 'common_counts' : 110, 'range_counts' : 3226}
]

LD4 = [
    {'ds_counts' : 443, 'dds_counts' : 191, 'common_counts' : 216, 'range_counts' : 4368},
    {'ds_counts' : 1712, 'dds_counts' : 504, 'common_counts' : 471, 'range_counts' : 4368},
    {'ds_counts' : 1324, 'dds_counts' : 464, 'common_counts' : 412, 'range_counts' : 4368}
]


f, axes = plt.subplots(ncols=2, nrows=2, sharey=True )

c1 = pd.DataFrame(LD1)
c1['types'] = pd.Series(['High Threshold', 'Low Threshold', 'MZ Threshold'])
c1 = c1[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts']]
c1.plot(x='types', kind='barh', stacked=True, title='LD1 Anomalies', mark_right=True)

c1['total'] = c1.sum(axis=1)


df_total = c1['total']
df_rel = c1[c1.columns[1:]].div(df_total, 0)*100

for n in df_rel:
    for i, (cs, ab, pc, tot) in enumerate(zip(c1.iloc[:, 1:].cumsum(1)[n], c1[n], df_rel[n], df_total)):
        
        if pc >= 15:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center')
        elif pc >= 3:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', rotation=90)

plt.show()

c2 = pd.DataFrame(LD2)
c2['types'] = pd.Series(['High Threshold', 'Low Threshold', 'MZ Threshold'])
c2 = c2[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts']]
c2.plot(x='types', kind='barh', stacked=True, title='LD2 Anomalies', mark_right=True)

c2['total'] = c2.sum(axis=1)
#c2 = c2[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts', 'total']]

df_total = c2['total']
df_rel = c2[c2.columns[1:]].div(df_total, 0)*100

for n in df_rel:
    for i, (cs, ab, pc, tot) in enumerate(zip(c2.iloc[:, 1:].cumsum(1)[n], c2[n], df_rel[n], df_total)):
        if pc >= 15:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center')
        elif pc >= 3:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', rotation=90)

plt.show()


c3 = pd.DataFrame(LD3)
c3['types'] = pd.Series(['High Threshold', 'Low Threshold', 'MZ Threshold'])
c3 = c3[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts']]
c3.plot(x='types', kind='barh', stacked=True, title='LD3 Anomalies', mark_right=True)

c3['total'] = c3.sum(axis=1)
#c3 = c3[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts', 'total']]

df_total = c3['total']
df_rel = c3[c3.columns[1:]].div(df_total, 0)*100

for n in df_rel:
    for i, (cs, ab, pc, tot) in enumerate(zip(c3.iloc[:, 1:].cumsum(1)[n], c3[n], df_rel[n], df_total)):
        if pc >= 15:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center')
        elif pc >= 3:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', rotation=90)

plt.show()

c4 = pd.DataFrame(LD4)
c4['types'] = pd.Series(['High Threshold', 'Low Threshold', 'MZ Threshold'])
c4 = c4[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts']]
c4.plot(x='types', kind='barh', stacked=True, title='LD4 Anomalies', mark_right=True)

c4['total'] = c4.sum(axis=1)
#c4 = c4[['types', 'dds_counts', 'ds_counts', 'common_counts', 'range_counts', 'total']]

df_total = c4['total']
df_rel = c4[c4.columns[1:]].div(df_total, 0)*100

for n in df_rel:
    for i, (cs, ab, pc, tot) in enumerate(zip(c4.iloc[:, 1:].cumsum(1)[n], c4[n], df_rel[n], df_total)):
        if pc >= 15:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center')
        elif pc >= 3:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', rotation=90)



plt.show()


############################################################
# twitter comparison preparation


def set_non_outliers(r):
    if r['state'] != 'OUTLIER':
        return 'OK'
    else:
        return 'OUTLIER'


# pre-process twitter data -> creating comparable structure
threshold = ['HT', 'LT', 'MZ']
for p in range(1,5):
    ld = 'LD{}'.format(p)
    
    for t in threshold:
        twitter_file = pd.read_csv('twitter_labels/{}_{}_twitter_label_672.csv'.format(ld, t))
        twitter_file.columns = ['oi_annot', 'oi', 'value']
        twitter_file['state'] = 'OUTLIER'
        twitter_file = twitter_file[['oi', 'state']]
        twitter_file['oi'] = twitter_file['oi'] - 1
        actual_ticks = pd.read_csv('xticks.csv', header=None)
        actual_ticks.columns = ['oi','time']
        full_annotation = pd.merge(actual_ticks, twitter_file, on='oi', how='left')
        full_annotation['state'] = full_annotation[['state']].apply(set_non_outliers, axis=1)
        full_annotation.to_csv('twitter_labels/{}_{}_twitter_672_annotation.csv'.format(ld, t))


