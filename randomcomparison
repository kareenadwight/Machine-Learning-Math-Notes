import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])

y_scores = np.array([45, 50, 55, 60, 70, 75, 85, 90])

reg = LinearRegression()
reg.fit(X, y_scores)

y_labels = (y_scores >= 60).astype(int)

clf = LogisticRegression()
clf.fit(X, y_labels)

X_test = np.array([[3.5], [6.5]])

print("Regression predictions:", reg.predict(X_test))
print("Classification predictions:", clf.predict(X_test))
print("Class probabilities:", clf.predict_proba(X_test))
