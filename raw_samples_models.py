#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:21:50 2018

@author: gciniwe
"""
import json, random
import numpy as np
import pandas as pd
import functions as f
from sklearn import preprocessing
from keras.optimizers import SGD, Adam, RMSprop


files #list containing full path for files containing raw segmented chirps

#create vector of labels
categ = [1]*5 + [2]*5  + [3]*5  + [4]*5  + [5]*5  + [6]*5  



samp = []
samp_sizes = []
for nams in files:
    temp = []
    with open(nams) as json_data:
        for line in json_data:
            x = json.loads(line) 
            y = f.butter_bandpass_filter(x, (44100/2)/2, 44100, order=5)
            y = y[0::2]
            samp_sizes.append(len(y))
            temp.append(y)
        samp.append(temp)    
del x, y, temp


#look at the distribution of the length of the clicks
#then choose a good cut-off point
import matplotlib.pyplot as plt
plt.hist(samp_sizes)
#choose 3000 samples!


#truncate all sequences longer than 3000 and pad all sequences shorter than 3000
Data = []
size = 3000
for clicks in samp:
	data = []
	for clck in clicks:
		if len(clck) >= size:
			data.append(clck[0:size])
		if len(clck) < size:
			ap = np.zeros(shape = (size-len(clck))  )
			data.append(np.concatenate((clck, ap)))
	Data.append(data)
    
del samp  

#split data into train, validate and testing sets 
train, validate, test = pd.DataFrame(), pd.DataFrame(),pd.DataFrame()

for count, lists in enumerate(Data):
    fn = 'recording_'+str(count)
    fn = pd.DataFrame(lists)
    y = pd.DataFrame(np.repeat(categ[count], fn.shape[0]))
    fn['y'] = y
    
    random.seed(123)
    train_data, validate_data, test_data = np.split(fn.sample(frac=1), [int(.6*len(fn)), int(.8*len(fn))])
    train = train.append(train_data); validate = validate.append(validate_data); test = test.append(test_data)
    

#data processing and reshaping
train = train.sample(frac=1)
validate = validate.sample(frac=1)
test = test.sample(frac=1)

X_train = train.iloc[:,0:3000]
y_train = train.iloc[:,3000]

    
X_validate = validate.iloc[:,0:3000]
y_validate = validate.iloc[:,3000]

X_test = test.iloc[:,0:3000]
y_test = test.iloc[:,3000]

encoder = preprocessing.LabelEncoder()

encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_train = y_train.reshape(-1, 1)

encoder.fit(y_validate)
y_validate = encoder.transform(y_validate)
y_validate = y_validate.reshape(-1, 1)

encoder.fit(y_test)
y_test = encoder.transform(y_test)
y_test = y_test.reshape(-1, 1)


one_hot_encoding = preprocessing.OneHotEncoder(sparse=False)

one_hot_encoding.fit(y_train)
Y_train = one_hot_encoding.transform(y_train)

one_hot_encoding.fit(y_validate)
Y_validate = one_hot_encoding.transform(y_validate)

one_hot_encoding.fit(y_test)
Y_test = one_hot_encoding.transform(y_test)


#covert to np array an reshape
X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_validate = np.array(X_validate)
X_validate = np.reshape(X_validate, (X_validate.shape[0], X_validate.shape[1], 1))        



#parameters for gridsearch
kernel_size = [3, 7]
dilation_rate = [1, 4]
batch_size = [64, 128, 256]    
lr = [0.001, 0.0001]
opt_ind = ['Adam', 'SGD', 'RMSProp']

  


#A. Convolutional Neural Network model
param_comb_conv = f.expandgrid(lr, batch_size, kernel_size, dilation_rate, opt_ind)

#train model and evaluate on validation set
for hp in param_comb_conv:
    	if hp[4]=='SGD':
    	    	opt = SGD(lr=hp[0])
    	if hp[4]=='Adam':
    	    	opt = Adam(lr=hp[0])
    	else:
    	    	opt = RMSprop(lr=hp[0])
      
    	model1 = f.conv_classifier(X_train, Y_train, 5, 128, kernel_size= hp[2], dilation_rate=hp[3],optimizer=opt)
    	history = model1[0].fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=50, batch_size=hp[1])
    	scores = model1[0].evaluate(X_validate, Y_validate, verbose=0, batch_size=hp[1])
    	print("Validation Accuracy: %2f%%" % (scores[1]*100))

    	file_id='/Users/gciniwe/Desktop/Final_Results/seq_results/conv_seq_'+str(hp[0])+"_"+str(hp[1])+"_"+str(hp[2])+"_"+str(hp[3])+"_"+str(hp[4])+'.txt'
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

best_model = f.conv_classifier(X_train, Y_train, 5, 128, kernel_size= kernel_size, dilation_rate=dilation_rate,optimizer=optimizer)
test_scores = best_model.evaluate(X_test, Y_test, verbose=0, batch_size=batch) 
print("Test Accuracy: %2f%%" % (test_scores[1]*100))



#B. Hybrid Convolutional and LSTM Neural Network model
param_comb_hybrid = f.expandgrid(lr, batch_size, kernel_size, opt_ind)
hp = param_comb_hybrid[0]
for hp in param_comb_hybrid:
    	if hp[3]=='SGD':
    	    	opt = SGD(lr=hp[0])
    	if hp[3]=='Adam':
    	    	opt = Adam(lr=hp[0])
    	else:
    	    	opt = RMSprop(lr=hp[0])
       
    	model1 = f.hybrid_classifier(X_train, Y_train, 5, kernel_size=hp[2], optimizer=opt)
    	history = model1[0].fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=50, batch_size=hp[1])
    	scores = model1[0].evaluate(X_validate, Y_validate, verbose=0, batch_size=hp[1])
    	print("Validation Accuracy: %2f%%" % (scores[1]*100))

    	file_id='/Users/gciniwe/Desktop/Final_Results/seq_results/hybrid_seq_'+str(hp[0])+"_"+str(hp[1])+"_"+str(hp[2])+"_"+str(hp[3])+'.txt'
    	with open(file_id, 'wb') as g:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            data = np.transpose(np.array([acc, val_acc, loss, val_loss]))
            np.savetxt(g, data, fmt=['%f','%f','%f','%f'])

#analyse results, then fit best model to test set, where best_ parameter are the values
#that obtained superior results on the validation set
batch = best_batch; kernel_size = best_kernel_size; optimizer = best_optimiser

best_model = f.hybrid_classifier(X_train, Y_train, 5, kernel_size=kernel_size, optimizer=optimizer)
test_scores = best_model.evaluate(X_test, Y_test, verbose=0, batch_size=batch) 
print("Test Accuracy: %2f%%" % (test_scores[1]*100))


