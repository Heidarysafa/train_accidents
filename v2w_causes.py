# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:02:55 2017

@author: mh4pk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 22:46:06 2017

@author: Moji
"""

from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import os

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model



from collections import Counter
from keras.callbacks import ModelCheckpoint
#import madules that contain implementation and evaluation of deep learning part
from ModelsAndEvaluation import build_model,load_best_model,evaluation_plot,save_history_plot,cross_val

trains = pd.read_csv('C:\\Users\\Moji\\Downloads\\from PC\\from PC\\total_reports6.csv', encoding='ISO-8859-1')
trains['cause_letter']= trains['cause_letter'].astype(str)
Counter(trains['cause_letter'])


import nltk
import gensim
naritives =trains['naritive'].tolist()

nar=[nltk.word_tokenize(sentences) for sentences in naritives ]

w2vmodel = gensim.models.Word2Vec(nar, size=100, window=5, min_count=5, workers=4)
len(w2vmodel.wv.vocab)
causes_categories = pd.Categorical(trains['cause_letter'])
causes_categories = causes_categories.codes

causes_categories.shape
labels = to_categorical((causes_categories))
labels.shape

selected_nar=nar
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




data = pad_sequences(narindexed, maxlen=500)

import random
indices = np.arange(data.shape[0])
random.seed(4)
random.shuffle(indices)
data = data[indices]
labels = labels[indices]
random.seed(4)
random.shuffle(naritives)

nb_validation_samples = int(0.1 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
naritive_val =naritives[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
'''
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
     data, labels, test_size=0.1, random_state=42)

print(y_train.shape)
'''

word_index = w2vmodel.wv.vocab

EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(w2vmodel.wv.vocab), 100))
for i in range(len(w2vmodel.wv.vocab)):
    embedding_vector = w2vmodel.wv[w2vmodel.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  
embedding_layer = Embedding(len(w2vmodel.wv.vocab),
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=500,
                            trainable=True)   
n= 5
####### CNN model from here ######## 
####################################
model =build_model("CNN",embedding_matrix,word_index,n, 'categorical_crossentropy')






model.summary()
#model_2 is going to be used to upload the loads for the best model during epoches
model_2=model
# checkpoint
filepath="cnn.w2v.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
cnn_history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=128,callbacks=callbacks_list, verbose=1)

model_2.load_weights("cnn.w2v.weights.best.hdf5")
model_2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



rnn_model  =build_model("RNN",embedding_layer,n)
rnn_model_2=rnn_model
# checkpoint
filepath="rnn.v2w.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
rnn_history= rnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=8, batch_size=128,callbacks=callbacks_list, verbose=1)

load_best_model(rnn_model_2,filepath)
rnn_model_2.loadweight("weights.best.hdf5")
rnn_model_2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
###################### evaluatins #############
evaluation_plot(rnn_model_2, x_val, y_val, ['E', 'H',  'M' ,'S' ,'T'], 'new_rnn_causes.pdf')



################ plots########
save_history_plot(rnn_history, 'rnn_history_causes_acc.pdf')

cross_val(data, labels,"CNN", embedding_matrix,word_index,n,'sparse_categorical_crossentropy', 10)

test_result= pd.DataFrame({'Narrative':naritive_val,'True_label': y_val_eva,'Predicted_label':y_pred })
# DF TO CSV
test_result.to_csv('Train_prediction_result.csv', sep=',')