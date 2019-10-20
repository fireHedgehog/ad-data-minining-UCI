from sklearn.neural_network import MLPClassifier
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def tuning_forest(x, y):
    clf = RandomForestClassifier(n_estimators=20)
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(5, 50),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    # run randomized search
    n_iter_search = 200
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


def tuning_tree(x, y):
    clf = DecisionTreeClassifier()
    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, None],
        # "max_features": sp_randint(2, 50),
        # "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
        # "min_impurity_decrease": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "min_samples_leaf": sp_randint(1, 20),
        "min_samples_split": sp_randint(2, 20),
        "presort": [True, False]
    }
    # run randomized search
    n_iter_search = 200
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


def tuning_l_svm(x, y):
    clf = SVC()
    param_dist = {
        "kernel": ["linear"],
        "C": [0.025, 0.1, 1, 5],
        "shrinking": [True, False],
        "probability": [True, False],
    }
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


def tuning_MPL(x, y):
    clf = MLPClassifier(activation='relu', solver='adam')
    param_dist = {
        "learning_rate_init": [0.001, 0.002, 0.025, 0.050, 0.01, 0.1, 1],
        "hidden_layer_sizes": [10, 50, 100, 150, 200, 250, 300, 350],
    }
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


def tuning_lo_reg(x, y):
    clf = LogisticRegression()
    param_dist = {
        # "penalty": ["l1", "l2"],
        "intercept_scaling": sp_randint(1, 200),
        "C": sp_randint(1, 200),
        "solver": ["liblinear"],
    }
    # run randomized search
    n_iter_search = 500
    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


def replace_missing_value(scale=''):
    data_frame = read_csv('../../static/missing_to_mean_ad_to_1_nonad_to_0.csv', low_memory=False)

    x = data_frame.iloc[:, 1:-1]  # do not read first column
    y = data_frame.iloc[:, -1]

    # n_row, n_column = data_frame.shape
    # print(n_row, n_column)

    if scale == "normal":
        nor_scaled = Normalizer()
        x = nor_scaled.fit_transform(x)
        return x, y
    elif scale == "standard":
        scaled = StandardScaler()
        x = scaled.fit_transform(x)
        return x, y

    return x, y


def feature_selection_method(x, y, method=''):
    if method == "SelectKBest":
        test = SelectKBest(chi2, k=1300)
        fit = test.fit(x, y)

        np.set_printoptions(precision=3)
        features = fit.transform(x)

        # summarize scores
        df_scores = DataFrame(fit.scores_)
        df_columns = DataFrame(x.columns)
        # concat two dataframes for better visualization
        feature_scores = concat([df_columns, df_scores], axis=1)
        feature_scores.columns = ['Features', 'Score']  # naming the dataframe columns

        # feature_scores.to_csv('../export/select_best_K.csv', encoding='utf-8')

        best_8 = feature_scores.nlargest(8, 'Score')
        # print(best_8)  # print 10 best features

        # plot_bar_for_best_k(best_8["Features"].values, best_8["Score"].values)
    return x


def plot_bar_for_best_k(label, values):
    # this is for plotting purpose
    print(label, values)
    index = np.arange(len(label))
    plt.bar(index, values)
    plt.xlabel('Features ', fontsize=5)
    plt.ylabel('Feature Score (Select Best K)', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Advertisement best 8 features ( SelectBestK() )')
    plt.show()


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

# X_train, X_test, Y_train, Y_test = train_test_split(x_selected, Y, train_size=0.66)
#
# # RandomOverSampler(return_indices=True)    # SMOTE()    #ADASYN()
# X_resampled, y_resampled = rebalancing(X_train, Y_train, SMOTE())

X_resampled, y_resampled = rebalancing(x_selected, Y, SMOTE())

# tuning_forest(X_resampled, y_resampled)
# tuning_tree(X_resampled, y_resampled)
# tuning_l_svm(X_resampled, y_resampled)
# tuning_MPL(X_resampled, y_resampled)
tuning_lo_reg(X_resampled, y_resampled)
