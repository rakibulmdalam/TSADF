import locale
import time
import datetime as d
import random
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def set_label_threshold(r, t1, t2):
    if (r['qd'] > t1) | (r['diff_qd'] > t2) | (r['wvs'] == 1000):
        return 'OUTLIER'
    else:
        return 'OK'


def set_label_baseline(r):
    if (r['qd_mz'] > 3.5) | (r['dqd_mz'] > 3.5) | (r['wvs'] == 1000):
        return 'OUTLIER'
    else:
        return 'OK'


def classify(x, what):
    if what == 'TP':
        if (x['label'] == 'OUTLIER') & (x['computed_label'] == 'OUTLIER'):
            return 1
        else:
            return 0

    elif what == 'TN':
        if (x['label'] == 'OK') & (x['computed_label'] == 'OK'):
            return 1
        else:
            return 0

    elif what == 'FP':
        if (x['label'] == 'OK') & (x['computed_label'] == 'OUTLIER'):
            return 1
        else:
            return 0

    elif what == 'FN':
        if (x['label'] == 'OUTLIER') & (x['computed_label'] == 'OK'):
            return 1
        else:
            return 0


def calculate_metrices(data):
    
    data['TP'] = data.apply(classify, args=('TP',), axis=1)
    data['TN'] = data.apply(classify, args=('TN',), axis=1)
    data['FP'] = data.apply(classify, args=('FP',), axis=1)
    data['FN'] = data.apply(classify, args=('FN',), axis=1)


    '''
    Sensitivity = TP / TP + FN
    Specificity = TN / TN + FP
    Precision = TP / TP + FP
    True-Positive Rate = TP / TP + FN
    False-Positive Rate = FP / FP + TN
    True-Negative Rate = TN / TN + FP
    False-Negative Rate = FN / FN + TP
    '''

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



def get_metrices(df, mz=True, t=()):
    if mz:
        df['computed_label'] = df.apply(set_label_baseline, axis=1)
    else:
        df['computed_label'] = df.apply(set_label_threshold, args=(t[0],t[1],), axis=1)
    
    data = pd.concat([df['label'],df['computed_label']], axis=1)
    return calculate_metrices(data)


def plot_anomalies(temp_df, diff_plot):
    # normal behavior calculation
    index = list()
    range_high = list()
    range_low = list()
    diff_range_high = list()
    diff_range_low = list()
    qd = list()
    diff_qd = list()
    for i in temp_df.index.values:
        index.append(i)
        range_high.append(temp_df.loc[i]['q3'])
        range_low.append(temp_df.loc[i]['q1'])
        diff_range_high.append(temp_df.loc[i]['diff_q3'])
        diff_range_low.append(temp_df.loc[i]['diff_q1'])
        qd.append(temp_df.loc[i]['qd'])
        diff_qd.append(temp_df.loc[i]['diff_qd'])


    _, ax = plt.subplots(figsize=(20,10))

    if diff_plot:
        ax.fill_between(index, diff_range_high, diff_range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
        temp_df['diff'].plot(ax=ax, color='mediumslateblue', label='differenced timeseries of vehicle count')
        s = temp_df[(temp_df['label'] == 'OK' ) & (temp_df['computed_label'] == 'OUTLIER') ]['diff']
        s2 = temp_df[(temp_df['label'] == 'OUTLIER' ) & (temp_df['computed_label'] == 'OK') ]['diff']
        anoms = temp_df[temp_df['computed_label'] == 'OUTLIER']['diff']
    else:
        ax.fill_between(index, range_high, range_low, color='lightblue', alpha=0.8, label='Normal Behavior (1st to 3rd quartile)')
        temp_df['value'].plot(ax=ax, color='mediumslateblue', label='timeseries of vehicle count')
        s = temp_df[(temp_df['label'] == 'OK' ) & (temp_df['computed_label'] == 'OUTLIER') ]['value']
        s2 = temp_df[(temp_df['label'] == 'OUTLIER' ) & (temp_df['computed_label'] == 'OK') ]['value']
        anoms = temp_df[temp_df['computed_label'] == 'OUTLIER']['value']

        print("FP")
        print(temp_df[(temp_df['label'] == 'OK' ) & (temp_df['computed_label'] == 'OUTLIER') ]['oi'])

        print("FN")
        print(temp_df[(temp_df['label'] == 'OUTLIER' ) & (temp_df['computed_label'] == 'OK') ]['oi'].values.tolist())

    ax.scatter(y=anoms.values, x=anoms.index.values, color='red', marker='x', s=300, label='all Anomalies')
    ax.scatter(y=s.values, x=s.index.values, color='darkblue', marker='$FP$', s=100, label='FP Anomalies')
    ax.scatter(y=s2.values, x=s2.index.values, color='darkgreen', marker='$FN$', s=200, label='FN Anomalies')
    ax.set_title('Anomalies FP and FN')
    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.grid(True)
    plt.show()



def do_it(omit_holidays=True):
    t1 = input('Point Distance Threshold: ')
    t2 = input('Difference Distance Threshold: ')
    plot_type = input('Original Data Plot (1) or Differenced Data Plot (2): ')
    
    thresholds = (int(t1), int(t2))
    df = pd.read_csv('ld2_with_labels_new.csv')


    #FNS = [101, 151, 172, 173, 262, 351, 352, 363, 364, 452, 454, 534, 636, 637, 665, 666, 667, 728, 735, 893, 894, 895, 896, 897, 899, 1001, 1088, 1113, 1121, 1225, 1232, 1276, 1297, 1504, 1577, 1578, 1701, 1808, 1878, 1889, 1951, 1952, 2005, 2197, 2202, 2373, 2473, 2493, 2648, 2667, 2668, 2670, 2761, 2833, 2854, 2859, 2921, 3107, 3125, 3126, 3138, 3313, 3342, 3531, 3536, 3682, 3683, 3721, 3834,3902, 4261, 4452, 4578, 6306, 6692, 6767, 6786, 6811, 6813, 7140, 7142, 7144, 7170, 7171, 7173, 7176, 7177, 7250, 7361, 7387, 7462, 7467, 7556, 7643, 7656, 7814, 7821, 7849, 7908, 7909, 7913, 7934, 7939, 7940, 7950, 7957, 7963, 8012, 8040, 8046, 8053, 8056, 8059, 8061, 8137, 8211, 8345, 8346, 8382, 8383, 8384, 8386, 8387, 8913, 8917, 8959, 8968, 9000, 9001, 9002, 9009, 9163, 9256, 9283,9285, 9364, 9375, 9554, 9555, 9572, 9636, 9642, 9667, 9729, 9758, 9840, 9862, 9928, 10028, 10417, 10433, 10435, 10437, 10491, 10531, 10533, 10626, 10627, 10628, 10629, 10705, 10900, 10933, 10939, 11086, 11089, 11096, 11105, 11183, 11197, 11201, 11203, 11209, 11294, 11297, 11376, 11405, 11454, 11466, 11473, 11679, 11680, 11681, 11682, 11683, 11684, 11685, 11853, 11854, 11977, 12032, 12037, 12048, 12151, 12152, 12153, 12154, 12155, 12319, 12320, 12333, 12334, 12335, 12336, 12337, 12338, 12339, 12343, 12346, 12357, 12415, 12419, 12420, 12424, 12425, 12426, 12427, 12511, 12527, 12571, 12608, 12628, 12801, 12818, 12992, 13027, 13118, 13276, 13280, 13281, 13282, 13283, 13284, 13285, 13286, 13316, 13375, 13591, 13657, 13731, 13732, 13853, 13855, 13856, 13857, 13974, 14065, 14079, 14188, 14189, 14261, 14338, 14340, 14549, 14873, 14978, 14979, 14980, 14981, 14982, 14983, 14984, 14985, 14986, 14987, 14988, 14989, 14990, 14991, 14992, 14993, 14994, 14995, 14996, 14997, 14998, 14999, 15000, 15490, 15866, 16118, 16121, 16345, 16449, 16474, 17113, 17308, 17317, 17428, 17567, 17571, 17572, 17573, 17641, 17730, 17814, 17915, 17937, 18020, 18499, 18575, 18584, 18585, 18607, 18611, 18752, 18804, 18889, 18896, 19233, 19328, 19349, 19354, 19749, 19828, 19902, 20100, 20210, 20382, 20418, 20478, 20481, 20706, 20720, 20762, 20783, 20787, 21026, 21188, 21272, 21444,22155, 22176, 22177, 22178, 22179, 22180, 22181, 22182, 22183, 22184, 22185, 22186, 22187, 22188, 22189, 22190, 22191, 22192, 22193, 22194, 22195, 22196, 22197, 22198, 22199, 22200, 22201, 22202, 22203, 22204, 22205, 22206, 22207, 22208, 22209, 22210, 22211, 22212, 22213, 22214, 22215, 22216, 22217, 22218, 22219, 22220, 22221, 22222, 22223, 22224, 22225, 22226, 22227, 22228, 22229, 22230, 22231, 22232, 22233, 22234, 22236, 22237, 22238, 22239, 22240, 22241, 22242, 22243, 22244, 22245, 22246, 22247, 22248, 22249, 22251, 22252, 22253, 22254, 22255, 22256, 22257, 22258, 22259, 22260, 22261, 22262, 22263, 22264, 22265, 22266, 22267, 22268, 22269, 22270, 22271, 22513, 22799, 22802, 22957, 22995, 23106, 23168, 23169, 23172, 23194, 23205, 23212, 23293, 23490, 23699, 23777, 23779, 23781, 23871, 23875, 23877, 23967, 23968, 24068, 24240, 24241, 24415, 24508, 24651, 24718, 24738, 24815, 24833, 24834, 24835, 24858, 24923, 25044, 25046, 25084, 25212, 25293, 25313, 25314, 25315, 25316, 25317, 25411, 25412, 25413, 25609, 25864, 25897, 25902, 26000, 26184, 26267, 26431, 26473, 26474, 26475, 26656, 26665, 26831, 26833, 26835, 26838, 26839, 26841, 26845, 26849, 26873, 26933, 26953, 27012, 27015, 27038, 27139, 27141, 27148, 27209, 27211, 27234, 27425, 27517, 27577, 27584, 27605, 27614, 27634, 27646, 27730, 27775, 27788, 27801, 27806, 27814, 27818, 27822, 27870,27871, 27967, 28003, 28005, 28072, 28073, 28099, 28108, 28159, 28185, 28432, 28549, 28577, 28586, 28591, 28766, 28768, 28840, 28841, 28890, 29003, 29026, 29033, 29048, 29062, 29331, 29747, 29748, 29790, 29815, 29913, 29999, 30015, 30111, 30167, 30295, 30303, 30585, 30654, 30655, 30692, 30766, 31135, 31273, 31359, 31557, 31658, 32129, 32131, 32403, 32404, 32511, 32698, 32699, 33066, 33078, 33148, 33253, 33255, 33342, 33343, 33345, 33531, 34248, 34250, 34836, 34921, 34922, 34933]

    if omit_holidays:
        metrices = get_metrices(df, True, thresholds) # mz = True/False    
    else:
        day_counter = 0
        df['date'] = df[['time']].apply(lambda x: str(x['time']).split(" ")[0], axis=1)

        for date in df['date'].unique().tolist():
            df.loc[df['date'] == date, 'doy'] = day_counter
            day_counter = day_counter + 1

        df['doy'] = df['doy'] + 1

        metrices = get_metrices(df, True, thresholds)
        holidays = [1, 6, 107, 112, 114, 115, 121, 153, 163, 164, 174, 227, 299, 305, 306, 331, 342, 359, 360, 365]
        for i in holidays:
            df.loc[df['doy'] == i, 'computed_label'] = 'OK'
            df.loc[df['doy'] == i, 'label'] = 'OK'
        data = pd.concat([df['label'],df['computed_label']], axis=1)
        metrices = calculate_metrices(data)
    
    
    acc = (metrices['TP'] + metrices['TN'])/ (metrices['TP'] + metrices['TN'] + metrices['FP'] + metrices['FN'])

    print(metrices)
    print('accuracy = {}'.format(acc))

    if int(plot_type) == 1:
        plot_anomalies(df, False) # diff = True/False whether to see differenced series or not
    else:
        plot_anomalies(df, True)



if __name__ == "__main__":
    choice = input('Process without holidays? (y / n) : ')
    if choice == 'y':
        do_it(False)
    else:
        do_it(True)



'''
    RECALL = TP / TP + FN
    Specificity = TN / TN + FP
    Precision = TP / TP + FP
    True-Positive Rate = TP / TP + FN
    False-Positive Rate = FP / FP + TN
    True-Negative Rate = TN / TN + FP
    False-Negative Rate = FN / FN + TP
'''



import random
i = random.sample(range(1, 595), 300)
FNS = [101, 151, 172, 173, 262, 351, 352, 363, 364, 452, 454, 534, 636, 637, 665, 666, 667, 728, 735, 893, 894, 895, 896, 897, 899, 1001, 1088, 1113, 1121, 1225, 1232, 1276, 1297, 1504, 1577, 1578, 1701, 1808, 1878, 1889, 1951, 1952, 2005, 2197, 2202, 2373, 2473, 2493, 2648, 2667, 2668, 2670, 2761, 2833, 2854, 2859, 2921, 3107, 3125, 3126, 3138, 3313, 3342, 3531, 3536, 3682, 3683, 3721, 3834,3902, 4261, 4452, 4578, 6306, 6692, 6767, 6786, 6811, 6813, 7140, 7142, 7144, 7170, 7171, 7173, 7176, 7177, 7250, 7361, 7387, 7462, 7467, 7556, 7643, 7656, 7814, 7821, 7849, 7908, 7909, 7913, 7934, 7939, 7940, 7950, 7957, 7963, 8012, 8040, 8046, 8053, 8056, 8059, 8061, 8137, 8211, 8345, 8346, 8382, 8383, 8384, 8386, 8387, 8913, 8917, 8959, 8968, 9000, 9001, 9002, 9009, 9163, 9256, 9283,9285, 9364, 9375, 9554, 9555, 9572, 9636, 9642, 9667, 9729, 9758, 9840, 9862, 9928, 10028, 10417, 10433, 10435, 10437, 10491, 10531, 10533, 10626, 10627, 10628, 10629, 10705, 10900, 10933, 10939, 11086, 11089, 11096, 11105, 11183, 11197, 11201, 11203, 11209, 11294, 11297, 11376, 11405, 11454, 11466, 11473, 11679, 11680, 11681, 11682, 11683, 11684, 11685, 11853, 11854, 11977, 12032, 12037, 12048, 12151, 12152, 12153, 12154, 12155, 12319, 12320, 12333, 12334, 12335, 12336, 12337, 12338, 12339, 12343, 12346, 12357, 12415, 12419, 12420, 12424, 12425, 12426, 12427, 12511, 12527, 12571, 12608, 12628, 12801, 12818, 12992, 13027, 13118, 13276, 13280, 13281, 13282, 13283, 13284, 13285, 13286, 13316, 13375, 13591, 13657, 13731, 13732, 13853, 13855, 13856, 13857, 13974, 14065, 14079, 14188, 14189, 14261, 14338, 14340, 14549, 14873, 14978, 14979, 14980, 14981, 14982, 14983, 14984, 14985, 14986, 14987, 14988, 14989, 14990, 14991, 14992, 14993, 14994, 14995, 14996, 14997, 14998, 14999, 15000, 15490, 15866, 16118, 16121, 16345, 16449, 16474, 17113, 17308, 17317, 17428, 17567, 17571, 17572, 17573, 17641, 17730, 17814, 17915, 17937, 18020, 18499, 18575, 18584, 18585, 18607, 18611, 18752, 18804, 18889, 18896, 19233, 19328, 19349, 19354, 19749, 19828, 19902, 20100, 20210, 20382, 20418, 20478, 20481, 20706, 20720, 20762, 20783, 20787, 21026, 21188, 21272, 21444,22155, 22176, 22177, 22178, 22179, 22180, 22181, 22182, 22183, 22184, 22185, 22186, 22187, 22188, 22189, 22190, 22191, 22192, 22193, 22194, 22195, 22196, 22197, 22198, 22199, 22200, 22201, 22202, 22203, 22204, 22205, 22206, 22207, 22208, 22209, 22210, 22211, 22212, 22213, 22214, 22215, 22216, 22217, 22218, 22219, 22220, 22221, 22222, 22223, 22224, 22225, 22226, 22227, 22228, 22229, 22230, 22231, 22232, 22233, 22234, 22236, 22237, 22238, 22239, 22240, 22241, 22242, 22243, 22244, 22245, 22246, 22247, 22248, 22249, 22251, 22252, 22253, 22254, 22255, 22256, 22257, 22258, 22259, 22260, 22261, 22262, 22263, 22264, 22265, 22266, 22267, 22268, 22269, 22270, 22271, 22513, 22799, 22802, 22957, 22995, 23106, 23168, 23169, 23172, 23194, 23205, 23212, 23293, 23490, 23699, 23777, 23779, 23781, 23871, 23875, 23877, 23967, 23968, 24068, 24240, 24241, 24415, 24508, 24651, 24718, 24738, 24815, 24833, 24834, 24835, 24858, 24923, 25044, 25046, 25084, 25212, 25293, 25313, 25314, 25315, 25316, 25317, 25411, 25412, 25413, 25609, 25864, 25897, 25902, 26000, 26184, 26267, 26431, 26473, 26474, 26475, 26656, 26665, 26831, 26833, 26835, 26838, 26839, 26841, 26845, 26849, 26873, 26933, 26953, 27012, 27015, 27038, 27139, 27141, 27148, 27209, 27211, 27234, 27425, 27517, 27577, 27584, 27605, 27614, 27634, 27646, 27730, 27775, 27788, 27801, 27806, 27814, 27818, 27822, 27870,27871, 27967, 28003, 28005, 28072, 28073, 28099, 28108, 28159, 28185, 28432, 28549, 28577, 28586, 28591, 28766, 28768, 28840, 28841, 28890, 29003, 29026, 29033, 29048, 29062, 29331, 29747, 29748, 29790, 29815, 29913, 29999, 30015, 30111, 30167, 30295, 30303, 30585, 30654, 30655, 30692, 30766, 31135, 31273, 31359, 31557, 31658, 32129, 32131, 32403, 32404, 32511, 32698, 32699, 33066, 33078, 33148, 33253, 33255, 33342, 33343, 33345, 33531, 34248, 34250, 34836, 34921, 34922, 34933]


df = pd.read_csv('ld2_with_labels.csv')
for e in i:
    print(FNS[e])
    df.loc[df['oi'] == FNS[e], "label"] = "OK"


df.to_csv('ld2_with_labels_new.csv')