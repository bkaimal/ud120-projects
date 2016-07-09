#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.svm import SVC
from sklearn import metrics


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

features_train = features_train[:len(features_train)]
labels_train = labels_train[:len(labels_train)]

clf = SVC(C=10000.0, kernel="rbf")

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
predictions = clf.predict(features_test)
score = metrics.accuracy_score(labels_test, predictions)
print "predict time:", round(time()-t1, 3), "s"

print("Accuracy: %f" % score)
print("predictions[10] : %d" % predictions[10])
print("predictions[26] : %d" % predictions[26])
print("predictions[50] : %d" % predictions[50])

num_chris = 0
for prediction in predictions:
    if prediction == 1:
        num_chris = num_chris + 1
print "Chris's emails : %d" % num_chris

#########################################################


