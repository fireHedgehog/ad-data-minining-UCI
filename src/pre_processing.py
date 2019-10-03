from pandas import read_csv, DataFrame
import matplotlib.pylab as plt
import numpy as np

ad_dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)

# drop all the missing values
ad_dataFrame = ad_dataFrame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
ad_dataFrame = ad_dataFrame.dropna()

# print first 10 columns to preview the data
print(ad_dataFrame.head(10))

# statistically describe it
# comment it out once we get the csv file.
# we can uncomment it to reprint.

# ------------------------------------------------------------------------------------------
# print(ad_dataFrame.describe())
# ad_dataFrame.describe().to_csv('../export/ad_data_describe.csv', encoding='utf-8')
# ------------------------------------------------------------------------------------------

# get the values from data frame
ad_array = ad_dataFrame.values

# separate data to feature data and class data
X = ad_array[:, 0:1558]  # feature
Y = ad_array[:, 1558]  # class

n_row, n_column = ad_dataFrame.shape
# print(n_row, n_column)

# comment it out once we get the csv file. since it is too slow to calculate
# print("\n correlation Matrix : ")
# print(ad_dataFrame.corr())

fig = plt.figure(1, figsize=(16, 9))
ax = fig.add_subplot(131)
# get the box plot of width and height
box_data = ad_dataFrame[['height', 'width']].values
print(box_data)
ax.boxplot(box_data, notch='True', patch_artist=True)
ax.set_xticklabels(['height', 'width'])
plt.show()
