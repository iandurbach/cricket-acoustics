#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:41:31 2018

@author: gciniwe
"""
import pandas as pd
import numpy
import itertools
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import functions as f

numpy.random.seed(7)

tempFeat #list containing full path for files containing the temporal features


X_train_call = []
X_test_call = []
X_validate_call =[]
Y_train_call = []
Y_test_call = []
Y_validate_call = []

#create vector of labels
categ = [1]*5 + [2]*5  + [3]*5  + [4]*5  + [5]*5  + [6]*5  

#split data into train, validate and test
for count, vec in enumerate(tempFeat):
    dat = pd.read_table(vec, sep=" ", header=None)   
    data = pd.DataFrame(preprocessing.scale(dat.ix[:,[1,3]])) 
    unik = dat.shape[0]
    #random.seed(123)
    sample = range(unik)
    train, validate, test = numpy.split(data.sample(frac=1), [int(.8*len(data)), int(.9*len(data))])
    
    R =  numpy.array(train)
    X_train_call.extend(R)
    Y_train_call.extend([categ[count]]*R.shape[0])

    S = numpy.array(test)
    X_test_call.extend(S)
    Y_test_call.extend([categ[count]]*S.shape[0])

    T = numpy.array(validate)
    X_validate_call.extend(T)
    Y_validate_call.extend([categ[count]]*T.shape[0])


X_train = numpy.array(X_train_call)
X_test = numpy.array(X_test_call)
X_validate = numpy.array(X_validate_call)

Y_train = numpy.array(Y_train_call)
Y_test = numpy.array(Y_test_call)
Y_validate = numpy.array(Y_validate_call)


    

min_samples_leaf = [1,5]
max_depth = [None, 15, 20, 50, 100]
min_samples_split = [2,5,10]

hyparams =  f.expandgrid(min_samples_leaf,max_depth,min_samples_split)

for hp in hyparams:
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=hp[1],min_samples_leaf=hp[0], min_samples_split = hp[2], criterion="entropy", max_features=None, oob_score=True, n_jobs=-1)
    
	
    #-------------------------training results---------------------------------------
    rf_classifier = rf_classifier.fit(X_train, Y_train)
    train_call_wise = rf_classifier.score(X_train, Y_train)

    #f_id = 'C:/Users/IBM_ADMIN/Downloads/Gciniwe/DataForResearch/DataForResearch/train_results_'+str(hp[0])+"_"+str(hp[1])+'.txt'
    f_id = '/Users/gciniwe/Documents/train_results_'+str(hp[0])+"_"+str(hp[1])+'.txt'
    with open(f_id, 'wb') as g:
        d = numpy.array([rf_classifier.oob_score_,train_call_wise ])
        numpy.savetxt(g, d, fmt='%f' )

	
    #------------------------validation results-------------------------------------
    validate_call_wise = rf_classifier.score(X_validate, Y_validate)

    #f_id = 'C:/Users/IBM_ADMIN/Downloads/Gciniwe/DataForResearch/DataForResearch/valid_results_'+str(hp[0])+"_"+str(hp[1])+ '.txt'
    f_id = '/Users/gciniwe/Documents/valid_results_'+str(hp[0])+"_"+str(hp[1])+'.txt'

    with open(f_id, 'wb') as i:
        d = numpy.array([validate_call_wise])
        numpy.savetxt(i, d, fmt='%f' )


#analyse results, then fit best model to test set, where best_ parameter are the values
#that obtained superior results on the validation set
min_samples_split = best_min_samples_split; min_samples_leaf = best_min_samples_leaf; max_depth = best_max_depth
best_rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=max_depth,min_samples_leaf=min_samples_leaf, min_samples_split = min_samples_split, criterion="entropy", max_features=None, oob_score=True, n_jobs=-1)
best_rf_classifier.fit(X_train, Y_train)
#-------------------------testing results---------------------------------------
#frame-wise accuracy score
test_score = best_rf_classifier.score(X_test, Y_test)
print('test_frame_wise',test_score )

