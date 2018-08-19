#!/usr/bin/python    
import sys
import pickle
import numpy
import sklearn
import matplotlib.pyplot
from time import time
from copy import copy
sys.path.append("../tools/")
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','total_payments','total_stock_value','exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
payment_features = ['total_payments', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
                   'deferral_payments', 'loan_advances', 'other', 'expenses', 
                   'director_fees']
stock_features = ['total_stock_value','exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred']
for name in data_dict:
    if data_dict[name]['total_payments'] >0 or data_dict[name]['total_stock_value'] >0:
        for feature in payment_features:
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0
        for feature in stock_features:
            if data_dict[name][feature] == 'NaN':
                data_dict[name][feature] = 0

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1


### store to my_dataset for easy export below
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Extract features and labels from dataset for local testing

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

parameters = {'criterion':('gini', 'entropy')
                }
DT = tree.DecisionTreeClassifier(random_state = 10)
clf = GridSearchCV(DT, parameters, scoring = 'f1')
clf= clf.fit(features_train, labels_train)
clf = clf.best_estimator_

estimators = [('scaler', MinMaxScaler()),
            ('reduce_dim', PCA()), 
            ('clf', clf)]
clf = Pipeline(estimators)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print "Accuracy: ", accuracy
target_names = ['non_poi', 'poi']
print classification_report(y_true = labels_test, y_pred =pred, target_names = target_names)

dump_classifier_and_data(clf, my_dataset, features_list)

