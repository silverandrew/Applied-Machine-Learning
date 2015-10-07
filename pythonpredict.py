import csv
import numpy as np
#from __future__ import print_function
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

f = open("data/hw2cleaneddata.csv")
data = np.loadtxt(fname = f, delimiter = ',')
#print(data)
X = data[:, 1:]  # select columns 1 through end
yf = data[:, 0]   # select column 0, the output label
#can only train on ints with linear SVM
y = yf.astype(int)
#print(y)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
  print("# Tuning hyper-parameters for %s" % score)
  print()

  clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
  clf.fit(X_train, y_train)

  print("Best parameters set found on development set:")
  print()
  print(clf.best_estimator_)
  print()
  print("Grid scores on development set:")
  print()
  for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std() / 2, params))
  print()

  print("Detailed classification report:")
  print()
  print("The model is trained on the full development set.")
  print("The scores are computed on the full evaluation set.")
  print()
  y_true, y_pred = y_test, clf.predict(X_test)
  print(classification_report(y_true, y_pred))
  print()

#might need to do this inside loop
clf.score(X_train, y_train)

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.

