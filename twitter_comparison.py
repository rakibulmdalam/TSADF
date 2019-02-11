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


############ pre processing ##############
def set_non_outliers(r):
    if r['state'] != 'OUTLIER':
        return 'OK'
    else:
        return 'OUTLIER'


# pre-process twitter data -> creating comparable structure
twitter_file = pd.read_csv('LD1_twitter_label_96.csv')

twitter_file.columns = ['oi_annot', 'oi', 'value']
twitter_file['state'] = 'OUTLIER'
twitter_file = twitter_file[['oi', 'state']]
twitter_file['oi'] = twitter_file['oi'] - 1
actual_ticks = pd.read_csv('xticks.csv', header=None)
actual_ticks.columns = ['oi','time']
full_annotation = pd.merge(actual_ticks, twitter_file, on='oi', how='left')
full_annotation['state'] = full_annotation[['state']].apply(set_non_outliers, axis=1)
full_annotation.to_csv('LD1_full_twitter_96_annotation.csv')


##### compare against manual labels ##########


##### integrate both labels together
# load manual labels 
manual_labels = pd.read_csv('full_annotation.csv')
manual_labels = manual_labels[['oi', 'time', 'value', 'state']]
# load twitter labels
twitter_labels = pd.read_csv('LD1_full_twitter_672_annotation.csv')
twitter_labels = twitter_labels[['oi', 'time', 'state']]

# include twitter label in manual_labels for comparison
manual_labels['twitter_state'] = twitter_labels['state']


#######  confusion matrix ########

def classify(x, what):
    if what == 'TP':
        if (x['state'] == 'OUTLIER') & (x['twitter_state'] == 'OUTLIER'):
            return 1
        else:
            return 0

    elif what == 'TN':
        if (x['state'] == 'OK') & (x['twitter_state'] == 'OK'):
            return 1
        else:
            return 0

    elif what == 'FP':
        if (x['state'] == 'OK') & (x['twitter_state'] == 'OUTLIER'):
            return 1
        else:
            return 0

    elif what == 'FN':
        if (x['state'] == 'OUTLIER') & (x['twitter_state'] == 'OK'):
            return 1
        else:
            return 0


def calculate_metrices(data):
    
    data['TP'] = data.apply(classify, args=('TP',), axis=1)
    data['TN'] = data.apply(classify, args=('TN',), axis=1)
    data['FP'] = data.apply(classify, args=('FP',), axis=1)
    data['FN'] = data.apply(classify, args=('FN',), axis=1)

    TP = len(data[data['TP'] == 1])
    FP = len(data[data['FP'] == 1])
    TN = len(data[data['TN'] == 1])
    FN = len(data[data['FN'] == 1])

    RECALL = TP / (TP + FN)
    SPECIFICITY = TN / (TN + FP)
    PRECISION = TP / (TP + FP)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)

    F1SCORE = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)

    m = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'RECALL': RECALL, 'PRECISION': PRECISION, 'SPECIFICITY': SPECIFICITY, 'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR, 'F1SCORE': F1SCORE}

    return m



def get_metrices(df):
    data = pd.concat([df['state'],df['twitter_state']], axis=1)
    return calculate_metrices(data)



metrices = get_metrices(manual_labels)
acc = (metrices['TP'] + metrices['TN'])/ (metrices['TP'] + metrices['TN'] + metrices['FP'] + metrices['FN'])
print(metrices)
print('accuracy = {}'.format(acc))



from ConfusionMatrix import ConfusionMatrix as CM


manual_labels = pd.read_csv('ld2_with_labels_new.csv')
 
manual_labels.rename(columns={'label':'state'}, inplace=True)
manual_labels = manual_labels[['oi', 'time', 'value', 'state']]
threshold = ['HT', 'LT', 'MZ']
for p in range(2,3):
    ld = 'LD{}'.format(p)
    for t in threshold:
        # load twitter labels
        twitter_labels = pd.read_csv('twitter_labels/{}_{}_twitter_672_annotation.csv'.format(ld, t))
        twitter_labels = twitter_labels[['oi', 'time', 'state']]

        # include twitter label in manual_labels for comparison
        manual_labels['twitter_state'] = twitter_labels['state']
        data = pd.concat([manual_labels['state'],manual_labels['twitter_state']], axis=1)
        con_mat = CM(data)
        print(con_mat.get())


#twitter vs ht

twitter_ld1 = {
    'ACCURACY': 0.9661244292237443,
    'TP': 1471, 
    'FP': 536, 
    'TN': 32382, 
    'FN': 651, 
    'ACCURACY': 0.9661244292237443, 
    'RECALL': 0.6932139491046183, 
    'PRECISION': 0.7329347284504235, 
    'SPECIFICITY': 0.9837171152560908, 
    'TPR': 0.6932139491046183, 
    'FPR': 0.01628288474390911, 
    'TNR': 0.9837171152560908, 
    'FNR': 0.3067860508953817, 
    'F1SCORE': 0.7125211915718092
}

ht_ld1 = {
            'ACCURACY': 0.9851312785388128,
            'TP': 1781, 
            'FP': 180, 
            'TN': 32738, 
            'FN': 341, 
            'RECALL': 0.8393025447690857, 
            'PRECISION': 0.9082100968893422, 
            'SPECIFICITY': 0.9945318670636126, 
            'TPR': 0.8393025447690857, 
            'FPR': 0.005468132936387387, 
            'TNR': 0.9945318670636126, 
            'FNR': 0.16069745523091422, 
            'F1SCORE': 0.8723977467548371
}


N = 4
ht = (ht_ld1['ACCURACY']*100,ht_ld1['RECALL']*100,ht_ld1['PRECISION']*100,ht_ld1['F1SCORE']*100)
fig, ax = plt.subplots()
ind = np.arange(N)    # the x locations for the groups
width = 0.1         # the width of the bars
p1 = ax.bar(ind, ht, width, color='#3D59AB')

twitter = (twitter_ld1['ACCURACY']*100,twitter_ld1['RECALL']*100,twitter_ld1['PRECISION']*100,twitter_ld1['F1SCORE']*100)
p2 = ax.bar(ind + width, twitter, width, color='#00CD00')

ax.set_title('Performance metrices')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Accuracy', 'Recall', 'Precision', 'F-Measure'))

ax.legend((p1[0], p2[0]), ('High tolerance threshold', 'Twitter Anomaly Detection'), loc='lower center')
ax.grid(True)
ax.autoscale_view()

plt.show()



ht_ld2 = {
            'TP': 2341, 
            'FP': 771, 
            'TN': 31633, 
            'FN': 295, 
            'ACCURACY': 0.9695776255707762,
            'RECALL': 0.8880880121396054, 
            'PRECISION': 0.7522493573264781, 
            'SPECIFICITY': 0.9762066411554129, 
            'TPR': 0.8880880121396054, 
            'FPR': 0.02379335884458709, 
            'TNR': 0.9762066411554129, 
            'FNR': 0.11191198786039454, 
            'F1SCORE': 0.8145441892832288
        }


twitter_ld2 = {
                    'TP': 1658, 
                    'FP': 391, 
                    'TN': 32013, 
                    'FN': 978, 
                    'ACCURACY': 0.9609303652968036, 
                    'RECALL': 0.6289833080424886,
                    'PRECISION': 0.8091752074182528, 
                    'SPECIFICITY': 0.9879335884458709, 
                    'TPR': 0.6289833080424886, 
                    'FPR': 0.01206641155412912, 
                    'TNR': 0.9879335884458709, 
                    'FNR': 0.37101669195751136, 
                    'F1SCORE': 0.7077908217716117
                }



N = 4
ht = (ht_ld2['ACCURACY']*100,ht_ld2['RECALL']*100,ht_ld2['PRECISION']*100,ht_ld2['F1SCORE']*100)
fig, ax = plt.subplots()
ind = np.arange(N)    # the x locations for the groups
width = 0.1         # the width of the bars
p1 = ax.bar(ind, ht, width, color='#3D59AB')

twitter = (twitter_ld2['ACCURACY']*100,twitter_ld2['RECALL']*100,twitter_ld2['PRECISION']*100,twitter_ld2['F1SCORE']*100)
p2 = ax.bar(ind + width, twitter, width, color='#00CD00')

ax.set_title('Performance metrices')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Accuracy', 'Recall', 'Precision', 'F-Measure'))

ax.legend((p1[0], p2[0]), ('High tolerance threshold', 'Twitter Anomaly Detection'), loc='lower center')
ax.grid(True)
ax.autoscale_view()

plt.show()
