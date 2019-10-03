from pandas import read_csv
import matplotlib.pylab as plt
import numpy as np

ad_dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)

# drop all the missing values
ad_dataFrame = ad_dataFrame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
ad_dataFrame = ad_dataFrame.dropna()

# print first 10 columns to preview the data
print(ad_dataFrame.head(10))
# print the type
print(type(ad_dataFrame.head(2).values[1, 1]))

# statistically describe it
# comment it out once we get the csv file. since it is too slow to calculate
# we can uncomment it to reprint.

# ------------------------------------------------------------------------------------------
# print(ad_dataFrame.describe())
# ad_dataFrame.describe().to_csv('../export/ad_data_describe.csv', encoding='utf-8')
# n_row, n_column = ad_dataFrame.shape
# print(n_row, n_column)
# print("\n correlation Matrix : ")
# print(ad_dataFrame.corr())
# ------------------------------------------------------------------------------------------


fig = plt.figure(1, figsize=(16, 9))
ax = fig.add_subplot(111)
# get the box plot of width and height
box_data = ad_dataFrame[['height', 'width']].values
box_data = box_data.astype(int)
ax.boxplot(box_data, notch='True', patch_artist=True)
ax.set_xticklabels(['height', 'width'])
plt.show()
