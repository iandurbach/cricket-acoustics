#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:53:19 2018

@author: gciniwe
"""

import random
import numpy as np
import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop
from sklearn import preprocessing
import functions as f
np.random.seed(7)


from sklearn.ensemble import RandomForestClassifier


fullFeatVectors #list containing full path of files with MFCC feature matrices

#create vector of labels
categ = [1]*5 + [2]*5  + [3]*5  + [4]*5  + [5]*5  + [6]*5  


frame_lengths = []
for count, vec in enumerate(fullFeatVectors):
    dat = pd.read_table(vec, sep=" ")
    frame_lengths.append(dat.shape[0])


#look at the distribution of the number of frames
import matplotlib.pyplot as plt
plt.hist(frame_lengths)
#choose 200 frames!



X_train_chirp = []
Y_train_chirp = []

X_validate_chirp =[]
Y_validate_chirp = []

X_test_chirp = []
Y_test_chirp = []


X_train_frame = pd.DataFrame()
X_validate_frame = pd.DataFrame()
X_test_frame = pd.DataFrame()


X_train_chirp_unaltered = []
X_test_chirp_unaltered = []
X_validate_chirp_unaltered = []



frame_no = 200


#split data into train, validate and test sets for two scenarios
#1. One scenario is for the random forest models that in the frames as they are
#2. Another scenario is for the NN model that require fixed length inputs. It is for this case that 
#restriction for 200 frames was made for.
     
for count, vec in enumerate(fullFeatVectors):
    dat = pd.read_table(vec, sep=" ")   
    temp_ind  = np.reshape( np.array(dat.iloc[:,39]), (dat.shape[0], 1)    )
    data = pd.DataFrame(np.concatenate((preprocessing.scale(dat.iloc[:,0:39]), temp_ind ), axis=1) )
    y = pd.DataFrame(np.repeat(categ[count], data.shape[0]))
    data['y'] = y
    
    sample = pd.DataFrame(np.unique(data.iloc[:,39]))
    random.seed(123)

    train_ind, val_ind, test_ind = np.split(sample.sample(frac=1), [int(.6*len(sample)), int(.8*len(sample))])
    
    train_ind = np.concatenate(np.array(train_ind), axis=0)
    val_ind = np.concatenate(np.array(val_ind), axis=0)
    test_ind = np.concatenate(np.array(test_ind), axis=0)

    trainData = data.loc[data.iloc[:,39].isin(train_ind)]
    testData = data.loc[data.iloc[:,39].isin(test_ind)]
    valData = data.loc[data.iloc[:,39].isin(val_ind)]
    
    sel = [1,2,3,4,5,6,7,8,9,10,11,12,13,40]
    
    X_train_frame = X_train_frame.append(trainData.iloc[:,sel])
    X_validate_frame = X_validate_frame.append(valData.iloc[:,sel])
    X_test_frame = X_test_frame.append(testData.iloc[:,sel])
    
    for tr in train_ind:
        R =  (trainData.loc[trainData.iloc[:,39]==tr]).iloc[:,1:14]
        X_train_chirp_unaltered.append(np.array(R))
        Y_train_chirp.append(categ[count])      
        if R.shape[0] >= frame_no:
            X_train_chirp.append(np.array(R.iloc[0:frame_no,]) )
        else:
            app = np.zeros(((frame_no-R.shape[0]), R.shape[1]))
            temp = np.append(R, app, axis=0)
            X_train_chirp.append(np.array(temp))

    for te in test_ind:
        S = (testData.loc[testData.iloc[:,39]==te]).iloc[:,1:14]
        X_test_chirp_unaltered.append(np.array(S))
        Y_test_chirp.append(categ[count])       
        if S.shape[0] >= frame_no:
            X_test_chirp.append(np.array(S.iloc[0:frame_no,]) )
        else:
            app = np.zeros(((frame_no-S.shape[0]), S.shape[1]))
            temp = np.append(S, app, axis=0)
            X_test_chirp.append(np.array(temp))

    for tv in val_ind:
        T = (valData.loc[valData.iloc[:,39]==tv]).iloc[:,1:14]
        X_validate_chirp_unaltered.append(np.array(T))
        Y_validate_chirp.append(categ[count])       
        if T.shape[0] >= frame_no:
            X_validate_chirp.append(np.array(T.iloc[0:frame_no,]) )
        else:
            app = np.zeros(((frame_no-T.shape[0]), T.shape[1]))
            temp = np.append(T, app, axis=0)
            X_validate_chirp.append(np.array(temp) )




#chirp level data processing and reshaping: NN
encoder = preprocessing.LabelEncoder()

encoder.fit(Y_train_chirp)
Y_train_chirp = encoder.transform(Y_train_chirp)
Y_train_chirp = Y_train_chirp.reshape(-1, 1)


encoder.fit(Y_test_chirp)
Y_test_chirp = encoder.transform(Y_test_chirp)
Y_test_chirp = Y_test_chirp.reshape(-1, 1)

encoder.fit(Y_validate_chirp)
Y_validate_chirp = encoder.transform(Y_validate_chirp)
Y_validate_chirp = Y_validate_chirp.reshape(-1, 1)


one_hot_encoding = preprocessing.OneHotEncoder(sparse=False)

one_hot_encoding.fit(Y_train_chirp)
Y_train_chirp_oh = one_hot_encoding.transform(Y_train_chirp)

one_hot_encoding.fit(Y_validate_chirp)
Y_validate_chirp_oh = one_hot_encoding.transform(Y_validate_chirp)

one_hot_encoding.fit(Y_test_chirp)
Y_test_chirp_oh = one_hot_encoding.transform(Y_test_chirp)


 
X_train_chirp = np.array(X_train_chirp)
X_test_chirp = np.array(X_test_chirp)
X_validate_chirp = np.array(X_validate_chirp) 




#parameters for gridsearch
kernel_size = [3, 7]
dilation_rate = [1, 4]
batch_size = [32, 64, 128, 256]    
lr = [0.01, 0.001, 0.0001]
opt_ind = ['Adam', 'SGD', 'RMSProp']

  


param_comb_conv = f.expandgrid(lr, batch_size, kernel_size, dilation_rate, opt_ind)

for hp in param_comb_conv:
    	if hp[4]=='SGD':
    	    	opt = SGD(lr=hp[0])
    	if hp[4]=='Adam':
    	    	opt = Adam(lr=hp[0])
    	else:
    	    	opt = RMSprop(lr=hp[0])

       
    	model1 = f.conv_classifier(X_train_chirp, Y_train_chirp_oh, filters=5, dense=50, kernel_size= hp[2], dilation_rate=hp[3],optimizer=opt)
    	history = model1[0].fit(X_train_chirp, Y_train_chirp_oh, validation_data=(X_validate_chirp, Y_validate_chirp_oh), epochs=50, batch_size=hp[1])
    	scores = model1[0].evaluate(X_validate_chirp, Y_validate_chirp_oh, verbose=0, batch_size=hp[1])
    	print("Validation Accuracy: %2f%%" % (scores[1]*100))

    	file_id='/Users/gciniwe/Desktop/Final_Results/mfcc_results/conv_seq_'+str(hp[0])+"_"+str(hp[1])+"_"+str(hp[2])+"_"+str(hp[3])+"_"+str(hp[4])+'.txt'
    	with open(file_id, 'wb') as g:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            data = np.transpose(np.array([acc, val_acc, loss, val_loss]))
            np.savetxt(g, data, fmt=['%f','%f','%f','%f'])


#analyse results, then fit best model to test set, where best_ parameter are the values
#that obtained superior results on the validation set
batch = best_batch; kernel_size = best_kernel_size; dilation_rate = best_dilation_rate; optimizer = best_optimiser

best_model = f.conv_classifier(X_train_chirp, Y_train_chirp_oh, 5, 128, kernel_size= kernel_size, dilation_rate=dilation_rate,optimizer=optimizer)
test_scores = best_model.evaluate(X_test_chirp, Y_test_chirp_oh, verbose=0, batch_size=batch) 
print("Test Accuracy: %2f%%" % (test_scores[1]*100))


param_comb_lstm = f.expandgrid(lr, batch_size, opt_ind)
for hp in param_comb_lstm:
    	if hp[3]=='SGD':
    	    	opt = SGD(lr=hp[0])
    	if hp[3]=='Adam':
    	    	opt = Adam(lr=hp[0])
    	else:
    	    	opt = RMSprop(lr=hp[0])

       
    	model1 = f.lstm_classifier(X_train_chirp, Y_train_chirp_oh, optimizer=opt)
    	history = model1[0].fit(X_train_chirp, Y_train_chirp_oh, validation_data=(X_validate_chirp, Y_validate_chirp_oh), epochs=50, batch_size=hp[1])
    	scores = model1[0].evaluate(X_validate_chirp, Y_validate_chirp_oh, verbose=0, batch_size=hp[1])
    	print("Validation Accuracy: %2f%%" % (scores[1]*100))

    	file_id='/Users/gciniwe/Desktop/Final_Results/mfcc_results/lstm_seq_'+str(hp[0])+"_"+str(hp[1])+'.txt'
    	with open(file_id, 'wb') as g:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            data = np.transpose(np.array([acc, val_acc, loss, val_loss]))
            np.savetxt(g, data, fmt=['%f','%f','%f','%f'])


#analyse results, then fit best model to test set, where best_ parameter are the values
#that obtained superior results on the validation set
batch = best_batch; kernel_size = best_kernel_size; dilation_rate = best_dilation_rate; optimizer = best_optimiser

best_model = f.lstm_classifier(X_train_chirp, Y_train_chirp_oh, optimizer=optimizer)
test_scores = best_model.evaluate(X_test_chirp, Y_test_chirp_oh, verbose=0, batch_size=batch) 
print("Test Accuracy: %2f%%" % (test_scores[1]*100))








#frame level data processing and reshaping: RF
Y_train_frame = X_train_frame.iloc[:,13]            
Y_test_frame = X_test_frame.iloc[:,13]            
Y_validate_frame = X_validate_frame.iloc[:,13]

encoder = preprocessing.LabelEncoder()
encoder.fit(Y_train_frame)
Y_train_frame = encoder.transform(Y_train_frame)
Y_train_frame = np.ravel(Y_train_frame.reshape(-1, 1))

Y_test_frame = encoder.transform(Y_test_frame)
Y_test_frame = np.ravel(Y_test_frame.reshape(-1, 1))

Y_validate_frame = encoder.transform(Y_validate_frame)
Y_validate_frame = np.ravel(Y_validate_frame.reshape(-1, 1))

X_train_frame = X_train_frame.iloc[:,0:13]
X_validate_frame = X_validate_frame.iloc[:,0:13]
X_test_frame = X_test_frame.iloc[:,0:13]
            
 

max_features = ['sqrt', 0.8, None]
max_depth = [10, 20, None]
min_samples_leaf = [1, 10, 20]  

hyparams_rf =  f.expandgrid(max_features, max_depth, min_samples_leaf)

for hp in hyparams_rf:
    rf_classifier = RandomForestClassifier(n_estimators=100 , criterion="entropy", min_samples_leaf=hp[2], max_features=hp[0], oob_score=True, n_jobs=-1, max_depth=hp[1])
    rf_classifier = rf_classifier.fit(X_train_frame, Y_train_frame) 

	
    #-------------------------training results---------------------------------------
    #frame-wise accuracy score
    train_frame_wise = rf_classifier.score(X_train_frame, Y_train_frame)
    #call-wise accuracy score
    train_call_wise = f.evaluate(rf_classifier, X_train_chirp_unaltered, Y_train_chirp)

    f_id = '/Users/gciniwe/Desktop/Final_Results/mfcc_results/rf_train_results_'+str(hp[0])+"_"+str(hp[1])+'.txt'
    with open(f_id, 'wb') as r:
        d = np.array([rf_classifier.oob_score_, train_frame_wise, train_call_wise[0] ])
        np.savetxt(g, d, fmt='%f' )
    
    id = '/Users/gciniwe/Desktop/Final_Results/mfcc_results/rf_train_freq_table_' + str(hp[0]) + '_' + str(hp[1])+  '.txt'
    with open(id, 'w') as outfile:
        outfile.write(str(train_call_wise[1]))
	




    #------------------------validation results-------------------------------------
    #frame-wise accuracy score
    validate_frame_wise = rf_classifier.score(X_validate_frame, Y_validate_frame)
    #call-wise accuracy score
    validate_call_wise = f.evaluate(rf_classifier, X_validate_chirp_unaltered, Y_validate_chirp)

    f_id = '/Users/gciniwe/Desktop/Final_Results/mfcc_results/rf_valid_results_'+str(hp[0])+"_"+str(hp[1])+ '.txt'
    with open(f_id, 'wb') as t:
        d = np.array([validate_frame_wise, validate_call_wise[0] ])
        np.savetxt(t, d, fmt='%f' )


    id = '/Users/gciniwe/Desktop/Final_Results/mfcc_results/rf_validate_freq_table_' + str(hp[0]) + '_' + str(hp[1]) + '.txt'
    with open(id, 'w') as outfile:
        outfile.write(str(validate_call_wise[1]))




#analyse results, then fit best model to test set, where best_ parameter are the values
#that obtained superior results on the validation set
max_features = best_max_features; min_samples_leaf = best_min_samples_split; max_depth = best_max_depth
best_rf_classifier = RandomForestClassifier(n_estimators=100 , criterion="entropy", max_features=max_features, min_samples_leaf=min_samples_leaf, oob_score=True, n_jobs=-1, max_depth=max_depth)
best_rf_classifier.fit(X_train_frame, Y_train_frame)
#-------------------------testing results---------------------------------------
#frame-wise accuracy score
test_frame_wise = best_rf_classifier.score(X_test_frame, Y_test_frame)
print('test_frame_wise',test_frame_wise )
#call-wise accuracy score
test_call_wise = f.evaluate(rf_classifier, X_test_chirp_unaltered, Y_test_chirp)
print('test_chrip_wise',test_call_wise )

            
