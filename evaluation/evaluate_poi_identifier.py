#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here 
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)

i = 0
for i in range(len(predictions)):
    predictions[i] = 0.0

accuracy = metrics.accuracy_score(labels_test, predictions)

print("accuracy : %f" % accuracy)
print predictions
print labels_test
print "Num people : %d" % len(predictions)
ctr = 0
for prediction in predictions:
    if prediction == 1.0:
        ctr += 1

print "POIs : %d" % ctr
print "Precision : %f" % metrics.precision_score(labels_test, predictions)
print "Recall    : %f" % metrics.recall_score(labels_test, predictions)
