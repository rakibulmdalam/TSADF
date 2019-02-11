import numpy as np
import pandas as pd


class ConfusionMatrix:

    def __init__(self, df):
        self.set(df)
        self._create_matrix()

    def set(self, df):
        try:
            self.df = df
            self.df.columns = ['truth', 'test']
        except:
            print('DataFrame error. Provide a DataFrame with 2 Columns. Ground truth values should be in first')


    def get(self):
        return self.matrix

    def _create_matrix(self):
    
        self.df['TP'] = self.df.apply(self._classify, args=('TP',), axis=1)
        self.df['TN'] = self.df.apply(self._classify, args=('TN',), axis=1)
        self.df['FP'] = self.df.apply(self._classify, args=('FP',), axis=1)
        self.df['FN'] = self.df.apply(self._classify, args=('FN',), axis=1)

        TP = len(self.df[self.df['TP'] == 1])
        FP = len(self.df[self.df['FP'] == 1])
        TN = len(self.df[self.df['TN'] == 1])
        FN = len(self.df[self.df['FN'] == 1])

        RECALL = TP / (TP + FN)
        SPECIFICITY = TN / (TN + FP)
        PRECISION = TP / (TP + FP)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (FN + TP)

        F1SCORE = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
        ACCURACY = (TP + TN)/ (TP + TN + FP + FN)
        self.matrix = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'ACCURACY': ACCURACY, 'RECALL': RECALL, 'PRECISION': PRECISION, 'SPECIFICITY': SPECIFICITY, 'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR, 'F1SCORE': F1SCORE}


    def _classify(self, x, what):
        if what == 'TP':
            if (x['truth'] == 'OUTLIER') & (x['test'] == 'OUTLIER'):
                return 1
            else:
                return 0

        elif what == 'TN':
            if (x['truth'] == 'OK') & (x['test'] == 'OK'):
                return 1
            else:
                return 0

        elif what == 'FP':
            if (x['truth'] == 'OK') & (x['test'] == 'OUTLIER'):
                return 1
            else:
                return 0

        elif what == 'FN':
            if (x['truth'] == 'OUTLIER') & (x['test'] == 'OK'):
                return 1
            else:
                return 0

