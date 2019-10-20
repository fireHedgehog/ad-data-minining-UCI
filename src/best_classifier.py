from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd


def classify_by_x_y(x, y):
    model = AdaBoostClassifier(
        RandomForestClassifier(
            n_estimators=50,
            bootstrap=False,
            criterion='gini',
            max_features=5,
            min_samples_leaf=1,
            min_samples_split=6),
        algorithm='SAMME.R',
        n_estimators=50)

    y_pred = cross_val_predict(model, x, y, cv=10)

    matrix = confusion_matrix(y_true=y, y_pred=y_pred)

    print("Random Forest")
    print(classification_report(y, y_pred))
    print("\n Accuracy score: \n", accuracy_score(y_true=y, y_pred=y_pred))
    print("\n matrix: \n", matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.heatmap(matrix, annot=True, cmap="YlGnBu")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Ad', 'Non-ad'])
    ax.yaxis.set_ticklabels(['Ad', 'Non-ad'])
    plt.show()

    # labels, title and ticks


def replace_missing_value(scale=''):
    data_frame = read_csv('../static/missing_to_mean_ad_to_1_nonad_to_0.csv', low_memory=False)

    x = data_frame.iloc[:, 1:-1]  # do not read first column
    y = data_frame.iloc[:, -1]

    return x, y


def feature_selection_method(x, y, method=''):
    if method == "SelectKBest":
        test = SelectKBest(chi2, k=1300)
        fit = test.fit(x, y)

        np.set_printoptions(precision=3)
        features = fit.transform(x)
        return features

    return x


def rebalancing(x, y, sampler):
    rus = sampler
    X_resampled, y_resampled = rus.fit_sample(x, y)

    df = DataFrame(y_resampled)
    df.columns = ["class"]

    df2 = DataFrame(y.values)
    df2.columns = ["class"]

    print('Original dataset shape %s' % Counter(y))
    print('Resampled dataset shape %s' % Counter(y_resampled))

    return X_resampled, y_resampled


X, Y = replace_missing_value()

x_selected = feature_selection_method(X, Y, "SelectKBest")

X_resampled, y_resampled = rebalancing(X, Y, SMOTE())

classify_by_x_y(X_resampled, y_resampled)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# sns.heatmap([[2781, 39], [39, 2781]], annot=True, cmap="YlGnBu")
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(['Ad', 'Non-ad'])
# ax.yaxis.set_ticklabels(['Ad', 'Non-ad'])
# plt.show()
