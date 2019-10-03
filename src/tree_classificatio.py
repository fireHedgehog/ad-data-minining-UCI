from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pandas import read_csv, DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ad_dataFrame = read_csv('../static/raw_data/ad_data.csv', low_memory=False)

# drop all the missing values
ad_dataFrame = ad_dataFrame.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
ad_dataFrame = ad_dataFrame.dropna()

# Define a standard scaler
sc = StandardScaler()

# Remove the first column, it's useless
data = ad_dataFrame.iloc[:, 1:].reset_index(drop=True)

# Factorization
data.loc[data['class'] == 'ad.', 'class'] = 1
data.loc[data['class'] == 'nonad.', 'class'] = 0

# Scale features and extract targets
# then get the values from data frame

x = data.iloc[:, :-1]
x = DataFrame(sc.fit_transform(x), index=x.index, columns=x.columns)
y = data.iloc[:, -1]

# ad_array = ad_dataFrame.values
# # separate data to feature data and class data
# X = ad_array[:, 0:1558]  # feature
# Y = ad_array[:, 1558]  # class

# random_state=1
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# set a baseline by DecisionTree.

classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)

y_pre_test = classifier.predict(X_test)
# cross validation to improve my model
scores_test = classifier.score(X_test, y_test)
print("\nTest Accuracy: {0:.1f}%".format(np.mean(scores_test) * 100))

# print("\ntest Predication: \n", y_pre_test)
# print('decision_function:\n', clf.decision_function(x_train))

matrix = confusion_matrix(y_true=y_test, y_pred=y_pre_test)
print(matrix)
print(classification_report(y_test, y_pre_test))

sns.heatmap(matrix, annot=True, cmap="YlGnBu")
plt.show()

# print("Accuracy score of Decision Tree:", accuracy_score(y_test, y_pre_test))
