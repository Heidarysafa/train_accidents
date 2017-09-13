# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:32:17 2017

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

trains = pd.read_csv('C:\\Users\\Moji\\total_reports5.csv', encoding='ISO-8859-1')
count=0
for index, row in trains.iterrows():
    if row['CAUSE']=='H307':
        count+=1
        row['CAUSE']='H306'
        
print(count)
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


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(0.2 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

GLOVE_DIR = "/Users/Moji/Downloads/"
embeddings_index = {}
f = open( 'glove.6B.100d.txt',encoding="utf8")
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
print(embedding_matrix.shape)        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=500,
                            trainable=True)

####### CNN model from here ######## 
####################################


sequence_input = Input(shape=(500,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)


x = Conv1D(64, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x= Dropout(0.25)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x= Dropout(0.25)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = MaxPooling1D(15)(x)  # global max pooling
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x= Dropout(0.25)(x)
preds = Dense(n, activation='softmax')(x)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])





model.summary()

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=30, batch_size=128)


sequence_input = Input(shape=(500,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)




x=(GRU(32, return_sequences=True,  recurrent_dropout=0.2))(embedded_sequences)
x=Dropout(0.2)(x)
x = GRU(32,  recurrent_dropout=0.2)(x)
x=Dropout(0.2)(x)

#x= GRU(25, return_sequences=True)(x)
x = Dense(128, activation='relu')(x)
x= Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x= Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


print("model fitting - attention GRU network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=7, batch_size=128)
###################### evaluatins #############
np.argmax([[2,5,4,3,2,1,1],[0,1,2,3,2,1,0]],axis=1)
y_pred=model.predict(x_val)
y_pred= np.argmax(y_pred,axis=1)
y_val=np.argmax(y_val,axis=1)
from sklearn.metrics import f1_score
f1_score(y_val, y_pred,average='micro')
import itertools
import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if normalize:
      #  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     #   print("Normalized confusion matrix")
    #else:
     #   print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
from sklearn.metrics import confusion_matrix   
cnf_matrix = confusion_matrix(y_val, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=['H018', 'H302',  'H307-6', 'H702', 'M302' ,'M405' ,'T110' ,'T220' ,'T314'],
                      title='Confusion matrix, without normalization')
plt.show()
  
  