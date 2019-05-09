import pandas as pd

import numpy as np
import os
path = os.path.dirname(__file__)
path1 = os.path.join(path, 'datasets/cleveland-normalize-binary.csv')


df = pd.read_csv(path1)


df = df.replace('?', np.nan)


col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df.columns = col


print("Cleveland data. Size={}\nNumber of missing values".format(df.shape))
print(df.isna().sum())



df=df.fillna(df.median())
df=df.fillna(df.median())
df=df.drop(['oldpeak', 'slope','ca', 'thal'], axis=1)
print("Concatanated dataset. Size={}\nNumber of missing values".format(df.shape))
print(df.isna().sum())

df.to_csv(os.path.join(path, 'recons_dataset/combined_dataset1.csv'), index=False)
