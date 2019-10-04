from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataFrame = read_csv('../static/missing_to_mean_ad_to_1_nonad_to_0.csv', low_memory=False)

# dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)
# # different missing value processing has, different accuracy
# dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)
# # drop all the missing values
# dataFrame = dataFrame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
# dataFrame = dataFrame.dropna()

x = dataFrame.iloc[:, :-1]
y = dataFrame.iloc[:, -1]

# scale the data
scaled = StandardScaler()
x = scaled.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.66)

models = [
    ('Logistic Regression ', LogisticRegression(solver='lbfgs')),
    ('GaussianNB', GaussianNB()),
    # n_neighbors=5
    ('K-Neighbors Classifier', KNeighborsClassifier()),
    ('Support Vector Machine', SVC(gamma=2, C=1)),
    ('Support Vector Machine Linear', SVC(kernel="linear", C=0.025)),
    # criterion='entropy', max_depth=150, min_samples_split=3
    ('Decision Tree Classifier', DecisionTreeClassifier(criterion='entropy')),
    # n_estimators=60
    ('Random Forest Classifier', RandomForestClassifier(n_estimators=10)),
]

for name, model in models:
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)

    print('\n model: {}\'s classificaition report is \n\n {}'.format(name, classification_report(y_predict, y_test)))
    print("\n Accuracy score: \n", accuracy_score(y_test, y_predict))
    print("\n matrix: \n", matrix)
