from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from collections import Counter


def classify_by_x_y(x_train, x_test, y_train, y_test):
    models = [
        ('Logistic Regression ', LogisticRegression(solver='liblinear', intercept_scaling=123, C=7)),
        ('Support Vector Machine Linear', SVC(kernel="linear", C=1)),
        ('Decision Tree Classifier',
         DecisionTreeClassifier(criterion='entropy',
                                max_features=43,
                                min_samples_leaf=1,
                                min_samples_split=2)),
        ('Random Forest Classifier',
         RandomForestClassifier(n_estimators=50,
                                bootstrap=False,
                                criterion='gini',
                                max_features=5,
                                min_samples_leaf=1,
                                min_samples_split=6)),
        ('Multilayer Perceptron', MLPClassifier())
    ]

    for name, model_ in models:
        model = BaggingClassifier(model_,
                                  n_estimators=50)

        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)

        print('\n model: {}\'s classificaition report is \n\n {}'.
              format(name, classification_report(y_predict, y_test)))
        print("\n Accuracy score: \n", accuracy_score(y_test, y_predict))
        print("\n matrix: \n", matrix)


def replace_missing_value(scale=''):
    data_frame = read_csv('../static/missing_to_mean_ad_to_1_nonad_to_0.csv', low_memory=False)

    x = data_frame.iloc[:, 1:-1]  # do not read first column
    y = data_frame.iloc[:, -1]

    # scaled = StandardScaler()
    # x = scaled.fit_transform(x)
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

X_train, X_test, Y_train, Y_test = train_test_split(x_selected, Y, train_size=0.66)

X_resampled, y_resampled = rebalancing(X_train, Y_train, SMOTE())

classify_by_x_y(X_resampled, X_test, y_resampled, Y_test)
