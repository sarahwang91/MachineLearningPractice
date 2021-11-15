#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:40:17 2021

@author: sarah
"""

from vecstack import stacking
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score #works
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from collections import Counter #for Smote, 

import warnings
warnings.filterwarnings("ignore")


#import dataset
trainfile = (r'/Users/sarah/Downloads/RevisedHomesiteTrain1.csv')
testfile = (r'/Users/sarah/Downloads/RevisedHomesiteTest1.csv')
trainData = pd.read_csv(trainfile)
testData = pd.read_csv(testfile)

print(trainData.shape)
print(trainData.head())    
print(testData.shape)
print(testData.head())


#Create demo train dataset
#d = trainData.iloc[:, :]


#Separate target column from demo data
#X_train = d.iloc[:, :-1].copy()
#Y_train = d.iloc[:, -1].copy()
#print(dx_train.shape)
#print(dy_train.shape)


#Check data types
print(trainData.info())
print(testData.info())


#List the name of all columns from a dataframe
TrainCols = list(trainData.columns.values)
TestCols = list(testData.columns.values)
print(TrainCols)
print(TestCols)


#Separate target column from train data
Xtrain = trainData.iloc[:, :-1].copy()
Ytrain = trainData.iloc[:, -1].copy()
print(Xtrain.shape)
print(Ytrain.shape)
#Copy data excluding categorical/duplicated column
Xtest = testData.iloc[:, :-1].copy()
print(Xtest.shape)


#Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size = .30, random_state = 1)
print(X_train.shape)
print(Y_train.shape)


#Pie plot of original dataset
print(Y_train.value_counts())
Y_train.value_counts().plot.pie(autopct = '%.2f')


#Use SMOTE to over-sampling minority data
print("___________________________________________________________________\nSMOTE\n")
print('Original dataset shape %s' % Counter(Y_train))
sm = SMOTE(sampling_strategy='not majority')
#sm = SMOTE(sampling_strategy='float', ratio=0.5)
X_res, Y_res = sm.fit_resample(X_train, Y_train)
print('Resampled dataset shape %s' % Counter(Y_res))


#Construct default decision tree and obtain respective accuracy with imbalanced data
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

Y_Pred = clf.predict(X_test)

print("Accuracy Score before SMOTE:", accuracy_score(Y_test, Y_Pred))
print("Confusion Matrix for Decision Tree before SMOTE")
print(confusion_matrix(Y_test, Y_Pred))
print("=== Classification Report before SMOTE ===")
print(classification_report(Y_test, Y_Pred))


#Run decision tree classifier after SMOTE
clf.fit(X_res, Y_res)
Y_Pred = clf.predict(X_test)

print("Accuracy Score after SMOTE:", accuracy_score(Y_test, Y_Pred))
print("Confusion Matrix for Decision Tree after SMOTE")
print(confusion_matrix(Y_test, Y_Pred))
print("=== Classification Report after SMOTE ===")
print(classification_report(Y_test, Y_Pred))


#Construct Random Forest Model
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_predict=rfc.predict(X_test)
print("accuracy Score (training) for RandomForest:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(Y_test,rfc_predict))


#Random Forest Model after SMOTE
rfc.fit(X_res, Y_res)
rfc_predict=rfc.predict(X_test)
print("accuracy Score (training) for RandomForest:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(Y_test,rfc_predict))


#Construct K-Nearest neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_predict=knn.predict(X_test)
print("accuracy Score (training) for KNN:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for KNN:")
print(confusion_matrix(Y_test,knn_predict))


#KNN after SMOTE
knn.fit(X_res, Y_res)
knn_predict=knn.predict(X_test)
print("accuracy Score (training) for KNN:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for KNN:")
print(confusion_matrix(Y_test,knn_predict))


#Construct Multilayer Perceptron Classifier
mlp = MLPClassifier()
mlp.fit(X_train, Y_train)
mlp_predict=mlp.predict(X_test)
print("accuracy Score (training) for Multilayer Perceptron:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Multilayer Perceptron:")
print(confusion_matrix(Y_test,mlp_predict))


# MLP Classifier of SMOTE
mlp.fit(X_res, Y_res)
mlp_predict=mlp.predict(X_test)
print("accuracy Score (training) for Multilayer Perceptron:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Multilayer Perceptron:")
print(confusion_matrix(Y_test,mlp_predict))


#Construct Support Vector machines Classifier
svc = SVC()
svc.fit(X_train, Y_train)
svc_predict = svc.predict(X_test)
print("accuracy Score (training) for Suppert Vector Classifier:{0:6f}".format(svc.score(X_test, Y_test)))
print("Confusion Matrix for Suppert Vector Classifier:")
print(confusion_matrix(Y_test,svc_predict))


#SVC after SMOTE
svc.fit(X_res, Y_res)
svc_predict = svc.predict(X_test)
print("accuracy Score (training) for Suppert Vector Classifier:{0:6f}".format(svc.score(X_test, Y_test)))
print("Confusion Matrix for Suppert Vector Classifier:")
print(confusion_matrix(Y_test,svc_predict))


#Ensemble methods stacking
print("___________________________________________________________________________________________\nEnsemble Methods Predictions using DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, MLPClassifier, SVC\n")

models = [ DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier(), SVC() ]
      
S_Train, S_Test = stacking(models,                   
                           X_train, Y_train, X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=4, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


#Gradient Boosting Classifier
model = GradientBoostingClassifier()    
model = model.fit(S_Train, Y_train) #S_Train should be vaildation set
y_pred = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % accuracy_score(Y_test, y_pred))


#Hypertuning for ensemble model
parameters={'learning_rate':[0.01,.1]}
model_random = RandomizedSearchCV(model,parameters,n_iter=15)
model_random.fit(S_Train, Y_train)
grid_parm_model=model_random.best_params_
print(grid_parm_model)
model= GradientBoostingClassifier(**grid_parm_model)
model.fit(S_Train,Y_train)
model_predict = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % accuracy_score(Y_test, model_predict))


#Copy data excluding categorical/duplicated column
#Xtest = testData.iloc[:500, :-1].copy()
#print(Xtest.shape)


#Predict testData
clf_Xpred = clf.predict(Xtest)
rfc_Xpred = rfc.predict(Xtest)
knn_Xpred = knn.predict(Xtest)
mlp_Xpred = mlp.predict(Xtest)
svc_Xpred = svc.predict(Xtest)


#Stacking testData
stacked_test_predictions = np.column_stack((clf_Xpred, rfc_Xpred, knn_Xpred, mlp_Xpred, svc_Xpred))
model_Xpred = model.predict(stacked_test_predictions)
print(model_Xpred)


# Predict for test data
resultDataSet = pd.DataFrame()
resultDataSet['QuoteNumber'] = Xtest['QuoteNumber']
resultDataSet['QuoteConversion_Flag'] = model_Xpred
resultDataSet.head()


# Write results into a submission file
submission = pd.DataFrame({
    "QuoteNumber": resultDataSet['QuoteNumber'],
    "QuoteConversion_Flag": resultDataSet['QuoteConversion_Flag']
})

submission.to_csv('/Users/sarah/Downloads/submission.csv', header = True, index = False)
