# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 19:02:45 2018

@author: Moji
"""

import gensim
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import nltk
#import madules that contain implementation and evaluation of deep learning part
from ModelsAndEvaluation import build_model,load_best_model,evaluation_plot,save_history_plot,cross_val

trains = pd.read_csv('C:\\Users\\Moji\\Downloads\\from PC\\from PC\\total_reports6.csv', encoding='ISO-8859-1')
count=0
corpus= ""

pd.isnull(trains['naritive'])
naritives =trains['naritive'].tolist()

nar=[nltk.word_tokenize(sentences) for sentences in naritives ]

w2vmodel = gensim.models.Word2Vec(nar, size=100, window=5, min_count=5, workers=4)
len(w2vmodel.wv.vocab)
trains.CAUSE[trains['CAUSE']=='H307']='H306'
causes=trains['CAUSE'].value_counts()
most_causes=causes[0:9]
n=9

lists = [[] for _ in range(n)]
for index, row in trains.iterrows():
    for i,value in zip(range(n),most_causes.index):
        if row['CAUSE'] == value:
            lists[i].append(row)
            
df=[]
for a_cause_list in lists:
    df.append( pd.DataFrame(a_cause_list,columns= a_cause_list[0].index))
Selected_Data= pd.concat(df)

selected_naritives =Selected_Data['naritive'].tolist()


selected_nar=[nltk.word_tokenize(sentences) for sentences in selected_naritives ]
word_indexes={}
for i in range(len(w2vmodel.wv.vocab)):
   word_indexes[ w2vmodel.wv.index2word[i]]=i 
narindexed=[]   
for a_nar in selected_nar:
    a_nar_indexed=[]
    for a_word in a_nar:
        if a_word in word_indexes.keys():
            a_nar_indexed.append(word_indexes[a_word])
        else:
            a_nar_indexed.append(0)
    narindexed.append(a_nar_indexed)




causes_categories = pd.Categorical(Selected_Data['CAUSE'])
causes_categories = causes_categories.codes
labels = to_categorical((causes_categories))




data = pad_sequences(narindexed, maxlen=500)

'''
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(0.1 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
'''
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
     data, labels, test_size=0.1, random_state=42)
EMBEDDING_DIM =100
embedding_matrix = np.zeros((len(w2vmodel.wv.vocab), 100))
for i in range(len(w2vmodel.wv.vocab)):
    embedding_vector = w2vmodel.wv[w2vmodel.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector 

word_index = w2vmodel.wv.vocab
n=9 

####### CNN model from here ######## 
####################################       
cnn_model =build_model("CNN",embedding_matrix,word_index,n, 'categorical_crossentropy')

cnn_model.summary()
#model_2 is going to be used to upload the loads for the best model during epoches
cnn_model_2=cnn_model
# checkpoint
filepath="cnn.w2v.detail.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
cnn_history = cnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=128,callbacks=callbacks_list, verbose=1)
# using checkpoint list and filepath for weights load the best model during the training
load_best_model(cnn_model_2,filepath)
###################### evaluatins #############
evaluation_plot(cnn_model_2, x_val, y_val, ['H018', 'H302',  'H307-6', 'H702', 'M302' ,'M405' ,'T110' ,'T220' ,'T314'], 'detail_cnn_causes_v2w.pdf')


################ plots########
save_history_plot(cnn_history, 'cnn_history_detail_causes_acc_v2w.pdf')

cross_val(data, labels,"CNN", embedding_matrix,word_index,n,'sparse_categorical_crossentropy', 10)

####### RNN model from here ######## 
#################################### 


rnn_model  =build_model("RNN",embedding_matrix,word_index,n, 'categorical_crossentropy')
rnn_model_2=rnn_model
# checkpoint
filepath="rnn.v2w.detail.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
rnn_history= rnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=8, batch_size=128,callbacks=callbacks_list, verbose=1)

load_best_model(rnn_model_2,filepath)

###################### evaluatins #############
evaluation_plot(rnn_model_2, x_val, y_val, ['H018', 'H302',  'H307-6', 'H702', 'M302' ,'M405' ,'T110' ,'T220' ,'T314'], 'detail_rnn_causes_v2w.pdf')


################ plots########
save_history_plot(rnn_history, 'rnn_history_detail_causes_acc_v2w.pdf')

cross_val(data, labels,"RNN", embedding_matrix,word_index,n,'sparse_categorical_crossentropy', 10)
