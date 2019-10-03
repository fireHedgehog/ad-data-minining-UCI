from pandas import read_csv, DataFrame
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


def replace_missing(df):
    for i in df:
        df[i] = df[i].replace('[?]', np.NAN, regex=True).astype('float')
        df[i] = df[i].fillna(df[i].mean())
    return df


dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)

# drop all the missing values
ad_dataFrame = dataFrame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
ad_dataFrame = ad_dataFrame.dropna()

# print first 10 columns to preview the data
print(ad_dataFrame.head(10))
# print the type
print(type(ad_dataFrame.head(2).values[1, 1]))

# statistically describe it
# comment it out once we get the csv file. since it is too slow to calculate
# we can uncomment it to reprint.

# ------------------------------------------------------------------------------------------
# statistic_data = ad_dataFrame.values[:, 0:1558].astype(float)
# describe = DataFrame(statistic_data).describe()
# print(describe)
# describe.to_csv('../export/ad_data_describe_1.csv', encoding='utf-8')
# n_row, n_column = ad_dataFrame.shape
# print(n_row, n_column)
# print("\n correlation Matrix : ")
# print(describe.corr())
# describe.corr().to_csv('../export/ad_data_correlation_matrix.csv', encoding='utf-8')
# ------------------------------------------------------------------------------------------

# replace missing value with mean
# only visualize first 4 columns because they are not nominal
ad_missing_with_mean = replace_missing(dataFrame.iloc[:, [0, 1, 2, 3]].copy())
values = ad_missing_with_mean.values
# print(values)

# we can print box plot
# fig = plt.figure(1, figsize=(16, 9))
# ax = fig.add_subplot(111)
# box_data = ad_missing_with_mean[['height', 'width']].values
# ax.boxplot(box_data, notch='True', patch_artist=True)
# ax.set_xticklabels(['height', 'width'])
# plt.show()

# we can print distplot
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
# fig.set_figheight(9)
# fig.set_figwidth(16)
# sns.distplot(ad_missing_with_mean['height'], ax=ax1)
# sns.distplot(ad_missing_with_mean['width'], ax=ax2)
# sns.distplot(ad_missing_with_mean['aratio'], ax=ax3)
# sns.distplot(ad_missing_with_mean['local'], ax=ax4)

## g = sns.pairplot(data=ad_dataFrame.iloc[1:, [0, 1, 2, 3, 1558]], hue="class", markers=["o", "s"])
# g = sns.pairplot(data=ad_dataFrame.iloc[1:, [0, 1, 2, 3, 1558]])
# g.axes[0, 0].set_yticks([])
# g.axes[0, 0].set_xticks([])
# g.axes[0, 1].set_yticks([])
# g.axes[0, 1].set_xticks([])
# g.axes[0, 2].set_yticks([])
# g.axes[0, 2].set_xticks([])
# g.axes[0, 3].set_yticks([])
# g.axes[0, 3].set_xticks([])
#
# g.axes[1, 0].set_yticks([])
# g.axes[1, 0].set_xticks([])
# g.axes[1, 1].set_yticks([])
# g.axes[1, 1].set_xticks([])
# g.axes[1, 2].set_yticks([])
# g.axes[1, 2].set_xticks([])
# g.axes[1, 3].set_yticks([])
# g.axes[1, 3].set_xticks([])
#
# g.axes[2, 0].set_yticks([])
# g.axes[2, 0].set_xticks([])
# g.axes[2, 1].set_yticks([])
# g.axes[2, 1].set_xticks([])
# g.axes[2, 2].set_yticks([])
# g.axes[2, 2].set_xticks([])
# g.axes[2, 3].set_yticks([])
# g.axes[2, 3].set_xticks([])
#
# g.axes[3, 0].set_yticks([])
# g.axes[3, 0].set_xticks([])
# g.axes[3, 1].set_yticks([])
# g.axes[3, 1].set_xticks([])
# g.axes[3, 2].set_yticks([])
# g.axes[3, 2].set_xticks([])
# g.axes[3, 3].set_yticks([])
# g.axes[3, 3].set_xticks([])

plt.show()
