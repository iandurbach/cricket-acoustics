#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:15:42 2018

@author: gciniwe
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.metrics import accuracy_score
import numpy as np
import itertools
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	#high = highcut / nyq
	b, a = butter(order, low, btype='low')
	return b, a


def butter_bandpass_filter(data, lowcut, fs, order=5):
	b, a = butter_bandpass(lowcut, fs, order=order)
	y = lfilter(b, a, data)
	return y


def conv_classifier(X, y, filters, dense, kernel_size, dilation_rate, optimizer):
	model = Sequential()
	model.add(Conv1D(filters=filters, padding="same", activation='relu',input_shape=(X.shape[1], X.shape[2]), kernel_size=kernel_size, dilation_rate=dilation_rate) )
	model.add(MaxPooling1D(pool_size=3))    
	model.add(Conv1D(filters=filters, activation='relu', padding="same", kernel_size=kernel_size))
	model.add(MaxPooling1D(pool_size=3))
	model.add(Dropout(rate=0.25))
	model.add(Flatten())
	model.add(Dense(units = dense, activation='relu'))
	model.add(Dropout(rate = 0.5))
	model.add(Dense(activation='softmax', units = y.shape[1]))
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
	summary = model.summary()
	return model, summary


def hybrid_classifier(X, y, filters, kernel_size, optimizer):
	model = Sequential()
	model.add(Conv1D(filters=filters, padding="same", activation='relu', input_shape=(X.shape[1], X.shape[2]), kernel_size=kernel_size) )
	model.add(MaxPooling1D(pool_size=3))    
	model.add(LSTM(units = 10, return_sequences=True))
	model.add(TimeDistributed(Dense(units = 10)))
	model.add(Dropout(rate = 0.25))
	model.add(Flatten())
	model.add(Dropout(rate = 0.5))
	model.add(Dense( activation='softmax', units = y.shape[1]))
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
	summary = model.summary()
	return model, summary


def lstm_classifier(X, y, optimizer):
    model = Sequential()
    model.add(LSTM(units = 100, return_sequences=True, input_shape = (X.shape[1],  X.shape[2]) ) )
    model.add(LSTM(units = 32, return_sequences=True))
    model.add(LSTM(units = 32))
    model.add(Dropout(rate = 0.5))
    model.add(Dense( activation='softmax', units = y.shape[1]))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    summary = model.summary()
    return model, summary


def expandgrid(*itrs):
	product = list(itertools.product(*itrs))
	return product

def evaluate(classifier, X, Y_true):
    Y_pred = []
    freq_tab = []
    for example in X:
        predic = classifier.predict(example)
        b_count = np.bincount(predic)
        Y_pred.append(np.argmax(b_count))

        n_zero = np.nonzero(b_count)[0]
        freq_tab.append(list(zip(n_zero, b_count[n_zero])) )
               
    score = accuracy_score(Y_true, Y_pred)  
    return score , freq_tab   
