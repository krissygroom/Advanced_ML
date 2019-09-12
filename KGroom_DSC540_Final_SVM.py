# Scikit Template
"""created by Casey Bennett 2018, www.CaseyBennett.com
   Copyright 2018

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License (Lesser GPL) as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
"""

"""
Kristen Groom
Spring 2019
Final Project Section for SVM Model
Using Professor's Scikit template found at: http://www.caseybennett.com/teaching.html
DSC 540 Advanced Machine Learning, DePaul University
"""

import sys
import csv
import math
import numpy as np
from operator import itemgetter
import time
import sklearn

import plotly.plotly as py
import plotly.graph_objs as go

from IPython.display import IFrame  

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale

# Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


#############################################################################
#
# Global parameters
#
#####################

target_idx=0                                        # Index of Target variable
cross_val=1                                         # Control Switch for CV
norm_target=0                                       # Normalize target switch
norm_features=0                                     # Normalize target switch
binning=0                                           # Control Switch for Bin Target
bin_cnt=2                                           # If bin target, this sets number of classes
feat_select=1                                      # Control Switch for Feature Selection
fs_type=2                                           # Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         # Control switch for low variance filter on features
feat_start=1                                        # Start column of features
k_cnt=5                                             # Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3

# Set global model parameters
rand_st=0                                           # Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#####################

file1= csv.reader(open('movies_clean_norm.csv'), delimiter=',', quotechar='"')

# Read Header Line
header=next(file1)            

# Read data
data=[]
target=[]
for row in file1:
    # Load Target
    if row[target_idx]=='':                         # If target is blank, skip row
        continue
    else:
        target.append(float(row[target_idx]))       # If pre-binned class, change float to int

    # Load row into temp array, cast columns
    temp=[]
                 
    for j in range(feat_start,len(header)):
        if row[j]=='':
            temp.append(float())
        else:
            temp.append(float(row[j]))

    # Load temp into Data array
    data.append(temp)

  
#Test Print
print(header)
print(len(target),len(data))
'''for i in range(10):
    print(target[i])
    print(data[i])'''
print('\n')

data_np=np.asarray(data)
target_np=np.asarray(target)


#############################################################################
#
# Preprocess data
#
##########################################


if norm_target==1:
    #Target normalization for continuous values
    target_np=scale(target_np)

if norm_features==1:
    #Feature normalization for continuous values
    data_np=scale(data_np)

if binning==1:
    #Discretize Target variable with KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=[bin_cnt], encode='ordinal', strategy='quantile')                         # Strategy here is important, quantile creating equal bins, but kmeans prob being more valid "clusters"
    target_np_bin = enc.fit_transform(target_np.reshape(-1,1))

    #Get Bin min/max
    temp=[[] for x in range(bin_cnt+1)]
    for i in range(len(target_np)):
        for j in range(bin_cnt):
            if target_np_bin[i]==j:
                temp[j].append(target_np[i])

    for j in range(bin_cnt):
        print('Bin', j, ':', min(temp[j]), max(temp[j]), len(temp[j]))
    print('\n')

    #Convert Target array back to correct shape
    target_np=np.ravel(target_np_bin)



#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('--LOW VARIANCE FILTER ON--', '\n')
    
    #LV Threshold
    sel = VarianceThreshold(threshold=0.5)                                      #Removes any feature with less than 20% variance
    fit_mod=sel.fit(data_np)
    fitted=sel.transform(data_np)
    sel_idx=fit_mod.get_support()

    #Get lists of selected and non-selected features (names and indexes)
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)

    print('Selected', temp)
    print('Features (total, selected):', len(data_np[0]), len(temp))
    print('\n')

    #Filter selected columns from original dataset
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index


#Feature Selection
if feat_select==1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''
    
    print('--FEATURE SELECTION ON--', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        if binning==1:
            clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
            sel = RFE(clf, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
        if binning==0:
            rgr = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=3, criterion='mse', random_state=None)
            sel = RFE(rgr, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
            
        fit_mod=sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx=fit_mod.get_support()      

    if fs_type==2:
        #Wrapper Select via model
        if binning==1:
            clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
            sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
            print ('Wrapper Select - Random Forest: ')
        if binning==0:
            
            rgr = SVR(kernel='linear', gamma=0.1, C=1.0)
            sel = SelectFromModel(rgr, prefit=False, threshold='mean', max_features=None)
            print ('Wrapper Select: ')
            
        fit_mod=sel.fit(data_np, target_np)    
        sel_idx=fit_mod.get_support()

    if fs_type==3:       
        if binning==1:                                                              ######Only work if the Target is binned###########
            #Univariate Feature Selection - Chi-squared
            sel=SelectKBest(chi2, k=k_cnt)
            fit_mod=sel.fit(data_np, target_np)                                         #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
            print ('Univariate Feature Selection - Chi2: ')
            sel_idx=fit_mod.get_support()

        if binning==0:                                                              ######Only work if the Target is continuous###########
            #Univariate Feature Selection - Mutual Info Regression
            sel=SelectKBest(mutual_info_regression, k=k_cnt)
            fit_mod=sel.fit(data_np, target_np)
            print ('Univariate Feature Selection - Mutual Info: ')
            sel_idx=fit_mod.get_support()

        #Print ranked variables out sorted
        temp=[]
        scores=fit_mod.scores_
        for i in range(feat_start, len(header)):            
            temp.append([header[i], float(scores[i-feat_start])])

        print('Ranked Features')
        temp_sort=sorted(temp, key=itemgetter(1), reverse=True)                     #Doesn't always sort correctly (e.g. for Chi Sq), so doublecheck output
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np[0]), len(temp))
    print('\n')
            
                
    ##3) Filter selected columns from original dataset #########
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index)



#############################################################################
#
# Train SciKit Models
#
##########################################

print('--ML Model Output--', '\n')

#Test/Train split
data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)

####Classifiers####
if binning==1 and cross_val==0:    
    #SciKit Decision Tree
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=None)
    clf.fit(data_train, target_train)
    print('Decision Tree Acc:', clf.score(data_test, target_test))
    if bin_cnt<=2:                                                                                                  #AUC only works with binary classes, not multiclass
        print('Decision Tree AUC:', metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1]))             
    #joblib.dump(clf, 'DecTree_DSC540_HW1.pkl')                     #Save and pickle model

    #SciKit Random Forest
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
    clf.fit(data_train, target_train)
    print('Random Forest Acc:', clf.score(data_test, target_test))
    if bin_cnt<=2:                                                                                                  #AUC only works with binary classes, not multiclass
        print('Random Forest AUC:', metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1]))             
 
####Regressors####
if binning==0 and cross_val==0:
    #SciKit Decision Tree Regressor
    rgr = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=None)
    rgr.fit(data_train, target_train)
    print('Decision Tree RMSE:', math.sqrt(metrics.mean_squared_error(target_test, rgr.predict(data_test))))
    print('Decision Tree Expl Var:', metrics.explained_variance_score(target_test, rgr.predict(data_test)))

    #SciKit Random Forest Regressor
    rgr = RandomForestRegressor(n_estimators=500, max_features=.33, max_depth=None, min_samples_split=3, criterion='mse', random_state=None)
    rgr.fit(data_train, target_train)
    print('Random Forest RMSE:', math.sqrt(metrics.mean_squared_error(target_test, rgr.predict(data_test))))
    print('Random Forest Expl Var:', metrics.explained_variance_score(target_test, rgr.predict(data_test)))


####Cross-Val Classifiers####
if binning==1 and cross_val==1:
    #Setup Crossval classifier scorers
    if bin_cnt<=2:
        scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    else:
        scorers = {'Accuracy': 'accuracy'}
    
    #SciKit Decision Tree - Cross Val
    start_ts=time.time()
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=None)
    scores = cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
    scores_Acc = scores['test_Accuracy']
    print("Decision Tree Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))        
    if bin_cnt<=2:                                                                                                  #Only works with binary classes, not multiclass
        scores_AUC= scores['test_roc_auc']
        print("Decision Tree AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)

    #SciKit Random Forest - Cross Val
    start_ts=time.time()
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
    scores = cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
    scores_Acc = scores['test_Accuracy']
    print("Random Forest Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))        
    if bin_cnt<=2:                                                                                                  #Only works with binary classes, not multiclass
        scores_AUC= scores['test_roc_auc']
        print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)
    
    #SciKit Gradient Boosting - Cross Val
    start_ts=time.time()
    clf = GradientBoostingClassifier(n_estimators=100, loss = 'deviance', learning_rate=0.1, max_depth=3, min_samples_split=3, random_state=rand_st)
    scores = cross_validate(estimator = clf, X=data_np, y=target_np, scoring = scorers, cv=5)

    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("Gradient Boosting Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("Gradient Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)


    #SciKit Ada Boosting - Cross Val
    start_ts=time.time()
    clf = AdaBoostClassifier(n_estimators=100, base_estimator=None, learning_rate=0.1, random_state=rand_st)
    scores = cross_validate(estimator = clf, X=data_np, y=target_np, scoring = scorers, cv=5)

    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("Ada Boosting Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("Ada Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)


    #SciKit Neural Network - Cross Val
    start_ts=time.time()
    clf = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,), random_state=rand_st)
    scores = cross_validate(estimator = clf, X=data_np, y=target_np, scoring = scorers, cv=5)

    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("Neural Network Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("Neural Network AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)

####Function to use for plot###
def data_to_plotly(x):
    k = []
    
    for i in range(0, len(x)):
        k.append(x[i][0])
        
    return k


####Cross-Val Regressors####
if binning==0 and cross_val==1:
    #Setup Crossval regression scorers
    scorers = {'Neg_MSE': 'neg_mean_squared_error', 'expl_var': 'explained_variance'} 
    
    #SciKit SVM - Cross Val
    start_ts=time.time()
    rgr=SVR(kernel='linear', gamma=0.1, C=1.0)
    scores=cross_validate(estimator=rgr, X=data_np, y=target_np, scoring = scorers, cv=5)                                                                                                 

    scores_RMSE = np.asarray([math.sqrt(-x) for x in scores['test_Neg_MSE']])                                       #Turns negative MSE scores into RMSE
    scores_Expl_Var = scores['test_expl_var']
    print("SVM RMSE:: %0.2f (+/- %0.2f)" % ((scores_RMSE.mean()), (scores_RMSE.std() * 2)))
    print("SVM Expl Var: %0.2f (+/- %0.2f)" % ((scores_Expl_Var.mean()), (scores_Expl_Var.std() * 2)))
    print("CV Runtime:", time.time()-start_ts)
    
    ##other kernal models for visualization##
    rgr_rbf = SVR(kernel='rbf', gamma=0.1, C=1.0)
    
    y_lin = rgr.fit(data_train, target_train).predict(data_train) 
    y_rbf = rgr_rbf.fit(data_train, target_train).predict(data_train) 

### Visualization of SVR help found at https://plot.ly/scikit-learn/plot-svm-regression/####
    p1 = go.Scatter(x=data_to_plotly(data_train), y=target_train,
                    mode='markers',
                    marker=dict(color='darkorange'),
                    name='data')
    
    p2 = go.Scatter(x=data_to_plotly(data_train), y=y_lin, 
                mode='lines',
                line=dict(color='navy', width=2),
                name='RBF model')

    
    p3 = go.Scatter(x=data_to_plotly(data_train), y=y_rbf, 
                mode='lines',
                line=dict(color='cyan', width=2),
                name='Linear model')


    layout = go.Layout(title='Support Vector Regression',
                   hovermode='closest',
                   xaxis=dict(title='data'),
                   yaxis=dict(title='target')) 
    
    fig = go.Figure(data=[p1, p2, p3], layout=layout)
    py.plot(fig)
     

