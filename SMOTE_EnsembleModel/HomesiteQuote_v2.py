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
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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

'''
#Use SMOTE to over-sampling minority data
print("___________________________________________________________________\nSMOTE\n")
print('Original dataset shape %s' % Counter(Y_train))
sm = SMOTE(sampling_strategy='not majority')
#sm = SMOTE(sampling_strategy='float', ratio=0.5)
X_res, Y_res = sm.fit_resample(X_train, Y_train)
print('Resampled dataset shape %s' % Counter(Y_res))
'''

#Construct default decision tree and obtain respective accuracy with imbalanced data
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

Y_Pred = clf.predict(X_test)

print("accuracy Score (testing) for Decision Tree:{0:6f}:", accuracy_score(Y_test, Y_Pred))
print("Confusion Matrix for Decision Tree")
print(confusion_matrix(Y_test, Y_Pred))
print("=== Classification Report for Decision Tree ===")
print(classification_report(Y_test, Y_Pred))


#Hyperparameter tuning done for decision tree classifier
parameters={'min_samples_split' : range(10,100,10),'max_depth': range(1,20,2)}
clf_random = RandomizedSearchCV(clf,parameters,n_iter=15)
clf_random.fit(X_train, Y_train)
grid_parm=clf_random.best_params_
print(grid_parm)

#Using the parameters obtained from HyperParameterTuning in the DecisionTreeClassifier 
clf = DecisionTreeClassifier(**grid_parm)
clf.fit(X_train,Y_train)
clf_predict = clf.predict(X_test)


#Obtain accuracy ,confusion matrix,classification report and AUC values for the result above.
print("accuracy Score (testing) after hypertuning for Decision Tree:{0:6f}".format(clf.score(X_test,Y_test)))
print("Confusion Matrix after hypertuning for Decision Tree")
print(confusion_matrix(Y_test,clf_predict))
print("=== Classification Report after hypertuning for Decision Tree ===")
print(classification_report(Y_test,clf_predict))

#get cross-validation report
clf_cv_score = cross_val_score(clf, X_train, Y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(clf_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ",clf_cv_score.mean())

'''
#Run decision tree classifier after SMOTE
clf.fit(X_res, Y_res)
Y_Pred = clf.predict(X_test)

print("Accuracy Score after SMOTE:", accuracy_score(Y_test, Y_Pred))
print("Confusion Matrix for Decision Tree after SMOTE")
print(confusion_matrix(Y_test, Y_Pred))
print("=== Classification Report after SMOTE ===")
print(classification_report(Y_test, Y_Pred))
'''

#Construct Random Forest Model
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)
rfc_predict=rfc.predict(X_test)
print("accuracy Score (testing) for RandomForest:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(Y_test,rfc_predict))

#Hyperparameter tuning for random forest classifier
rfc_random = RandomizedSearchCV(rfc,parameters,n_iter=15)
rfc_random.fit(X_train, Y_train)
grid_parm_rfc=rfc_random.best_params_
print(grid_parm_rfc)

#Construct Random Forest with best parameters
rfc= RandomForestClassifier(**grid_parm_rfc)
rfc.fit(X_train,Y_train)
rfc_predict = rfc.predict(X_test)
print("accuracy Score (testing) after hypertuning for Random Forest:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix after hypertuning for Random Forest:")
print(confusion_matrix(Y_test,rfc_predict))
print("=== Classification Report after hypertuning for Random Forest ===")
print(classification_report(Y_test,rfc_predict))


#get cross-validation report
rfc_cv_score = cross_val_score(rfc, X_train, Y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ",rfc_cv_score.mean())

'''
#Random Forest Model after SMOTE
rfc.fit(X_res, Y_res)
rfc_predict=rfc.predict(X_test)
print("accuracy Score (training) for RandomForest:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(Y_test,rfc_predict))
'''


#Construct K-Nearest neighbors
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)
neigh_predict=neigh.predict(X_test)
print("accuracy Score (testing) for KNN:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for KNN:")
print(confusion_matrix(Y_test,neigh_predict))


#Hyperparameter tuning done for K-Nearest Neighbor classifier
parameters = {'n_neighbors':[3,5,7,9,11], 'weights':['uniform', 'distance'], 'p':[1,2]}


neigh_random = RandomizedSearchCV(neigh,parameters,n_iter=15)
neigh_random.fit(X_train, Y_train)
grid_parm=neigh_random.best_params_
print(grid_parm)

#Using the parameters obtained from HyperParameterTuning in the K-Nearest neighbors 
neigh = KNeighborsClassifier(**grid_parm)
neigh.fit(X_train,Y_train)
neigh_predict = neigh.predict(X_test)

#Obtain accuracy ,confusion matrix,classification report and AUC values for the result above.
print("accuracy Score (testing) after hypertuning for KNeighborsClassifier:{0:6f}".format(neigh.score(X_test,Y_test)))
print("Confusion Matrix after hypertuning for KNeighborsClassifier")
print(confusion_matrix(Y_test,neigh_predict))
print("=== Classification Report after hypertuning for KNeighborsClassifier ===")
print(classification_report(Y_test,neigh_predict))

#get cross-validation report
neigh_cv_score = cross_val_score(neigh, X_train, Y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(neigh_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - KNeighborsClassifier: ",neigh_cv_score.mean())

'''
#KNN after SMOTE
knn.fit(X_res, Y_res)
knn_predict=knn.predict(X_test)
print("accuracy Score (training) for KNN:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for KNN:")
print(confusion_matrix(Y_test,knn_predict))
'''

#Construct Multilayer Perceptron Classifier
mlp = MLPClassifier(max_iter=300, random_state=1)
mlp.fit(X_train, Y_train)
mlp_predict=mlp.predict(X_test)
print("accuracy Score (testing) for Multilayer Perceptron:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Multilayer Perceptron:")
print(confusion_matrix(Y_test,mlp_predict))

#Hyperparameter tuning done for MultiLayer Perceptron classifier

#parameters = {'hidden_layer_sizes':[(10,), (20,)], 'activation':['tanh', 'relu'], 'solver':['sgd', 'adam'], 'alpha': [0.0001, 0.05], 'learning_rate':['constant', 'adaptive']}
#parameters = {'hidden_layer_sizes':[(10,5), (20,5)], 'activation':['tanh', 'relu'], 'learning_rate':['constant', 'adaptive']}
#parameters = {'hidden_layer_sizes':[(10,5,3), (20,7,3)], 'activation':['tanh', 'relu'], 'learning_rate':['constant', 'adaptive'], 'max_iter' :[100, 150]}
parameters = {'hidden_layer_sizes':[(10,), (15,), (10,5), (20,7,3)]} #trys different layers, optimize one paramater then try others

mlp_random = RandomizedSearchCV(mlp,parameters,n_iter=15)
mlp_random.fit(X_train, Y_train)
grid_parm=mlp_random.best_params_
print(grid_parm)

#Using the parameters obtained from HyperParameterTuning in the MLPClassifier 
mlp = MLPClassifier(**grid_parm)
mlp.fit(X_train,Y_train)
mlp_predict = mlp.predict(X_test)

#Obtain accuracy ,confusion matrix,classification report and AUC values for the result above.
print("accuracy Score (testing) after hypertuning for MultiLayer Perceptron:{0:6f}".format(mlp.score(X_test,Y_test)))
print("Confusion Matrix after hypertuning for MultiLayer Perceptron")
print(confusion_matrix(Y_test,mlp_predict))
print("=== Classification Report after hypertuning for MultiLayer Perceptron ===")
print(classification_report(Y_test,mlp_predict))

#get cross-validation report
mlp_cv_score = cross_val_score(mlp, X_train, Y_train, cv=10, scoring="roc_auc")
print("=== All AUC Scores ===")
print(mlp_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - MultiLayer Perceptron: ",mlp_cv_score.mean())

'''
# MLP Classifier of SMOTE
mlp.fit(X_res, Y_res)
mlp_predict=mlp.predict(X_test)
print("accuracy Score (training) for Multilayer Perceptron:{0:6f}".format(rfc.score(X_test, Y_test)))
print("Confusion Matrix for Multilayer Perceptron:")
print(confusion_matrix(Y_test,mlp_predict))
'''

#Construct Support Vector machines Classifier
svc = SVC(max_iter=500)
svc.fit(X_train, Y_train)
svc_predict = svc.predict(X_test)
print("accuracy Score (testing) for Suppert Vector Classifier:{0:6f}".format(svc.score(X_test, Y_test)))
print("Confusion Matrix for Suppert Vector Classifier:")
print(confusion_matrix(Y_test,svc_predict))

'''
#SVC after SMOTE
svc.fit(X_res, Y_res)
svc_predict = svc.predict(X_test)
print("accuracy Score (testing) for Suppert Vector Classifier:{0:6f}".format(svc.score(X_test, Y_test)))
print("Confusion Matrix for Suppert Vector Classifier:")
print(confusion_matrix(Y_test,svc_predict))
'''

print("___________________________________________________________________\nSMOTE\n")
print('Original dataset shape %s' % Counter(Y_train))
sm = SMOTE(sampling_strategy='minority')
X_res, y_res = sm.fit_resample(X_train, Y_train)
print('Resampled dataset shape %s' % Counter(y_res))


#Ensemble methods stacking
'''clf = DecisionTreeClassifier({'min_samples_split': 70, 'max_depth': 9})
rfc = RandomForestClassifier({'min_samples_split': 30, 'max_depth': 19})
neigh = KNeighborsClassifier({'weights': 'uniform', 'p': 2, 'n_neighbors': 11})
mlp = MLPClassifier({'hidden_layer_sizes': (10, 5)})'''
print("___________________________________________________________________________________________\nEnsemble Methods Predictions using DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, MLPClassifier, SVC\n")

models = [ DecisionTreeClassifier(), RandomForestClassifier(),\
              KNeighborsClassifier(), MLPClassifier(), SVC() ]
      
S_Train, S_Test = stacking(models,                   
                           X_res, y_res, X_test,   
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

'''
#Gradient Boosting Classifier
model = GradientBoostingClassifier()    
model = model.fit(S_Train, Y_train) #S_Train should be vaildation set
y_pred = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % accuracy_score(Y_test, y_pred))
'''

#Random Forest Classifier
model = RandomForestClassifier()    
model = model.fit(S_Train, y_res) #S_Train should be vaildation set
y_pred = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % accuracy_score(Y_test, y_pred))

#Hyperparameter tuning for random forest classifier
parameters={'min_samples_split' : range(10,100,10),'max_depth': range(1,20,2)}
model_random = RandomizedSearchCV(model,parameters,n_iter=15)
model_random.fit(S_Train, y_res)
grid_parm_model=model_random.best_params_
print(grid_parm_model)
model = RandomForestClassifier(**grid_parm_model)
model.fit(S_Train,y_res)
model_predict = model.predict(S_Test)
print('Final prediction score for ensemble methods: [%.8f]' % accuracy_score(Y_test, model_predict))

'''
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
'''

#Copy data excluding categorical/duplicated column
#Xtest = testData.iloc[:500, :-1].copy()
#print(Xtest.shape)


#Predict testData
clf_Xpred = clf.predict(Xtest)
rfc_Xpred = rfc.predict(Xtest)
neigh_Xpred = neigh.predict(Xtest)
mlp_Xpred = mlp.predict(Xtest)
svc_Xpred = svc.predict(Xtest)


#Stacking testData
stacked_test_predictions = np.column_stack((clf_Xpred, rfc_Xpred, neigh_Xpred, mlp_Xpred, svc_Xpred))
model_Xpred = model.predict(stacked_test_predictions)
print(model_Xpred)

'''
# Predict for test data
resultDataSet = pd.DataFrame(model.predict_proba(S_Test))
resultDataSet['QuoteNumber'] = Xtest['QuoteNumber']
resultDataSet['QuoteConversion_Flag'] = model_Xpred
resultDataSet.head()
'''

#Get Prediction Probability for the predicted class as a dataframe
pred_Probability =pd.DataFrame(model.predict_proba(stacked_test_predictions))
pred_Probability.head()

pred_Probability['QuoteNumber'] = Xtest['QuoteNumber']
pred_Probability['QuoteConversion_Flag'] = pred_Probability.iloc[:, 1]



# Write results into a submission file
submission = pd.DataFrame({
    "QuoteNumber": pred_Probability['QuoteNumber'],
    "QuoteConversion_Flag": pred_Probability['QuoteConversion_Flag']
})

submission.to_csv('/Users/sarah/Downloads/submission_v2.csv', header = True, index = False)
