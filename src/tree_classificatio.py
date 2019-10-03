from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pandas import read_csv
import numpy as np

ad_dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)

# drop all the missing values
ad_dataFrame = ad_dataFrame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
ad_dataFrame = ad_dataFrame.dropna()

# get the values from data frame
ad_array = ad_dataFrame.values

# separate data to feature data and class data
X = ad_array[:, 0:1558]  # feature
Y = ad_array[:, 1558]  # class

# random_state=1
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)

# set a baseline by DecisionTree.

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("Accuracy score of Decision Tree:", accuracy_score(y_test, predictions))
