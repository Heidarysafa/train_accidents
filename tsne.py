# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:28:04 2017

@author: Moji
"""
import pandas as pd
import os
import numpy as np
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt

# read data 
trains = pd.read_csv('C:\\Users\\mh4pk\\Downloads\\total_reports5.csv', encoding='ISO-8859-1')
content=trains['naritive'].astype(str)
print(content)
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(content)
word_index = tokenizer.word_index

GLOVE_DIR = "C:\\Users\\mh4pk\\Downloads\\"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
        
        
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=250)       
reduced = tsne.fit_transform(embedding_matrix)

plt.figure(figsize=(200, 200), dpi=100)
max_x = np.amax(reduced, axis=0)[0]
max_y = np.amax(reduced, axis=0)[1]
plt.xlim((-max_x,max_x))
plt.ylim((-max_y,max_y))

plt.scatter(reduced[:, 0], reduced[:, 1], 20);

for row_id in range(0, len(word_index)):
    target_word = glove_words[word_index[row_id]]
    x = reduced[row_id, 0]
    y = reduced[row_id, 1]
    plt.annotate(target_word, (x,y))

plt.savefig("glove_2000.png");