# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:52:55 2018

@author: Moji
"""

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


#import madules that contain implementation and evaluation of deep learning part
from ModelsAndEvaluation import build_model,load_best_model,evaluation_plot,save_history_plot,cross_val

trains = pd.read_csv('C:\\Users\\Moji\\Downloads\\from PC\\from PC\\total_reports6.csv', encoding='ISO-8859-1')
count=0
corpus= ""

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


causes_categories = pd.Categorical(Selected_Data['CAUSE'])
causes_categories = causes_categories.codes
labels = to_categorical((causes_categories))

content=Selected_Data['naritive'].astype(str)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(content)

sequences = tokenizer.texts_to_sequences(content)
word_index = tokenizer.word_index


data = pad_sequences(sequences, maxlen=500)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
     data, labels, test_size=0.1, random_state=42)

GLOVE_DIR = "/Users/Moji/Downloads/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()



EMBEDDING_DIM = 100
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
####### CNN model from here ######## 
####################################
cnn_model =build_model("CNN",embedding_matrix,word_index,n, 'categorical_crossentropy')
model.summary()
#model_2 is going to be used to upload the loads for the best model during epoches
cnn_model_2=cnn_model
# checkpoint
filepath="cnn.glv.details.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
cnn_history = cnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=128,callbacks=callbacks_list, verbose=1)
# using checkpoint list and filepath for weights load the best model during the training
load_best_model(cnn_model_2,filepath)

###################### evaluatins #############
evaluation_plot(cnn_model_2, x_val, y_val, ['H018', 'H302',  'H307-6', 'H702', 'M302' ,'M405' ,'T110' ,'T220' ,'T314'], 'detail_cnn_causes_glv.pdf')


################ plots########
save_history_plot(cnn_history, 'cnn_history_detail_causes_acc_glv.pdf')

cross_val(data, labels,"CNN", embedding_matrix,word_index,n,'sparse_categorical_crossentropy', 10)

####### RNN model from here ######## 
####################################

rnn_model =build_model("RNN",embedding_matrix,word_index,n, 'categorical_crossentropy')
rnn_model.summary()
#model_2 is going to be used to upload the loads for the best model during epoches
rnn_model_2=rnn_model
# checkpoint
filepath="rnn.glv.detail.weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
rnn_history = rnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128,callbacks=callbacks_list, verbose=1)
# using checkpoint list and filepath for weights load the best model during the training
load_best_model(rnn_model_2,filepath)
###################### evaluatins #############
evaluation_plot(rnn_model_2, x_val, y_val, ['H018', 'H302',  'H307-6', 'H702', 'M302' ,'M405' ,'T110' ,'T220' ,'T314'], 'detail_rnn_causes_glv.pdf')


################ plots########
save_history_plot(rnn_history, 'rnn_history_detail_causes_acc_glv.pdf')

cross_val(data, labels,"RNN", embedding_matrix,word_index,n,'sparse_categorical_crossentropy', 6)
