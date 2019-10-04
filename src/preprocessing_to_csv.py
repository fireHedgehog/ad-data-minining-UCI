from pandas import read_csv
import numpy as np


def replace_missing(df):
    for i in df:
        df[i] = df[i].replace('[?]', np.NAN, regex=True).astype('float')
        df[i] = df[i].fillna(df[i].mean())
    return df


dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)

missing_to_mean = dataFrame
# 'height' 'width' 'aratio' 'local'
missing_to_mean[['height', 'width', 'aratio', 'local']] = replace_missing(missing_to_mean.iloc[:, [0, 1, 2, 3]].copy())

missing_to_mean.iloc[:, -1] = missing_to_mean.iloc[:, -1].replace(['ad.', 'nonad.'], [1, 0])

print(missing_to_mean)

# missing_to_mean.to_csv('../export/missing_to_mean_ad_to_1_nonad_to_0.csv', encoding='utf-8')
