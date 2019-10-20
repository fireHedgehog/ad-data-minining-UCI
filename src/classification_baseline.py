from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_selection import SelectKBest, RFECV, chi2
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from collections import Counter


def classify_by_x_y(x_train, x_test, y_train, y_test):
    models = [
        # ('Logistic Regression ', LogisticRegression(solver='lbfgs')),
        # ('GaussianNB', GaussianNB()),
        # # n_neighbors=5
        # ('K-Neighbors Classifier', KNeighborsClassifier()),
        # ('Support Vector Machine', SVC(gamma=2, C=1)),
        # ('Support Vector Machine Linear', SVC(kernel="linear", C=0.025)),
        # # criterion='entropy', max_depth=150, min_samples_split=3
        # ('Decision Tree Classifier', DecisionTreeClassifier()),
        # # n_estimators=60
        # ('Random Forest Classifier', RandomForestClassifier(n_estimators=10)),
        ('Multilayer Perceptron', MLPClassifier())
    ]

    for name, model in models:
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)

        print('\n model: {}\'s classificaition report is \n\n {}'.
              format(name, classification_report(y_predict, y_test)))
        print("\n Accuracy score: \n", accuracy_score(y_test, y_predict))
        print("\n matrix: \n", matrix)


def drop_missing_value():
    # different missing value processing has, different accuracy
    data_frame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)
    # drop all the missing values
    data_frame = data_frame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
    data_frame = data_frame.dropna()

    n_row, n_column = data_frame.shape
    print(n_row, n_column)

    x = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]
    return x, y


# get replaced data
# and normalize it
# if do not passing any param
# return raw X and Y
def replace_missing_value(scale=''):
    data_frame = read_csv('../static/missing_to_mean_ad_to_1_nonad_to_0.csv', low_memory=False)

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
        print(best_8)  # print 10 best features

        # plot_bar_for_best_k(best_8["Features"].values, best_8["Score"].values)

        return features

    elif method == "rfe":
        rf_model = RandomForestClassifier(max_depth=3, min_samples_leaf=2, n_estimators=10)
        rf_model.fit(x, y)
        rfe = RFECV(estimator=rf_model, step=1, cv=StratifiedKFold(10), scoring='accuracy')
        fit = rfe.fit(x, y)
        print('Optimal number of features: {}'.format(rfe.n_features_))

        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        f.set_figheight(10)
        f.set_figwidth(20)

        ax1.set_title('Recursive Feature Elimination (10-Cross-Validation)', fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlabel('Number of features selected', fontsize=14, labelpad=20)
        ax1.set_ylabel('% Correct Classification', fontsize=14, labelpad=20)
        ax1.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_, color='#303F9F', linewidth=3)

        df_imp = DataFrame(rfe.estimator_.feature_importances_)
        df_columns = DataFrame(x.columns)

        d_set = concat([df_columns, df_imp], axis=1)
        d_set.columns = ['Features', 'importance']
        d_set = d_set.sort_values(by='importance', ascending=False)

        ax2.barh(y=d_set['Features'], width=d_set['importance'], color='#1976D2')
        ax2.set_title('RFECV - Feature Importance', fontsize=20, fontweight='bold', pad=20)
        ax2.set_xlabel('Importance', fontsize=14, labelpad=20)

        plt.show()

        selected = fit.transform(x)
        DataFrame(selected).to_csv('../export/select_RFE.csv', encoding='utf-8')
        print(selected)

        return selected

    elif method == 'univariate':

        return x

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

    # f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    # ax1.set_title(" Original Dataset: ")
    # ax2.set_title(" Rebalanced Dataset: ")
    # sns.countplot(data=df2, y='class', palette='Set2', ax=ax1)
    # sns.countplot(data=df, y='class', palette='Set2', ax=ax2)
    # plt.show()

    print('Original dataset shape %s' % Counter(y))
    print('Resampled dataset shape %s' % Counter(y_resampled))

    return X_resampled, y_resampled


X, Y = replace_missing_value()

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.66)
x_selected = feature_selection_method(X, Y, "SelectKBest")

X_train, X_test, Y_train, Y_test = train_test_split(x_selected, Y, train_size=0.66)

# RandomOverSampler(return_indices=True)    # SMOTE()    #ADASYN()
X_resampled, y_resampled = rebalancing(X_train, Y_train, SMOTE())

classify_by_x_y(X_resampled, X_test, y_resampled, Y_test)
