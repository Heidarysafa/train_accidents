# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:46:05 2018

@author: Moji
"""
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt


EMBEDDING_DIM =100

def build_model(model,emb_matrix,word_index,n_output, loss_fun):
    '''This function will build a recurrent or convelution neural net with a predefined and teseted structure.
    model should be replaced by CNN or RNN as string emb_matrix is the pre-trained matrix for embedding,
    n_output number of output classes, loss_fun is the loss function such as categorical_crossentropy or sparse_categorical_crossentropy,
    
    '''
    if len(word_index)!=emb_matrix.shape[0]:
        layer_length = len(word_index)+1
        print('glove embedding')
        print(layer_length)
    else:
        layer_length = len(word_index)
        print('w2v embedding')
        
    emb_layer = Embedding(layer_length,
                            EMBEDDING_DIM,
                            weights=[emb_matrix],
                            input_length=500,
                            trainable=True)
    if model=="CNN":
        sequence_input = Input(shape=(500,), dtype='int32')
        embedded_sequences = emb_layer(sequence_input)


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
        preds = Dense(n_output, activation='softmax')(x)
        model = Model(sequence_input, preds)
        model.compile(loss=loss_fun,
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
    elif model=="RNN":
        sequence_input = Input(shape=(500,), dtype='int32')
        embedded_sequences = emb_layer(sequence_input)
    
        x=(GRU(64,  recurrent_dropout=0.2, return_sequences= True))(embedded_sequences)
        x=Dropout(0.2)(x)
        
        x=(GRU(64,  recurrent_dropout=0.2))(x)
        x=Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x= Dropout(0.5)(x)
        
        preds = Dense(n_output, activation='softmax')(x)
        
        rnn_model = Model(sequence_input, preds)
        rnn_model.compile(loss=loss_fun,
                      optimizer='adam',
                      metrics=['acc'])
        return rnn_model

def load_best_model (model, filepath):
    '''
    Loads the best function after training phase
    and compile it with adam optimizer and loss='categorical_crossentropy'
    '''
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def evaluation_plot(model_, x_val,y_val, classes, figure_format_name):
    '''
    plots confusion matrix for the test set and save it in a string named figure_format_name
    such as 'rnn_matrix.pdf' or 'cnn_conf_matrix.png'
    classes referes to a list of output classes as strings
    the deafult setting is for normalized matrices
    '''
    y_pred=model_.predict(x_val) # model should be replaced by best model i.e rnn_model_2 or cnn_model_2
    y_pred= np.argmax(y_pred,axis=1)
    y_val_eva=np.argmax(y_val,axis=1)
    
    
    
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
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
       
    cnf_matrix = confusion_matrix(y_val_eva, y_pred)
    np.set_printoptions(precision=2)
    fig =plt.figure()
    '''
    plot_confusion_matrix(cnf_matrix,classes= classes,
                          title='Confusion matrix, without normalization')
    plt.show()
    '''
    plot_confusion_matrix(cnf_matrix,classes=classes,
                         normalize =True, title='Confusion matrix, with normalization')
    plt.show()
      
    fig.savefig(figure_format_name, bbox_inches='tight')  

    # f1-macro is caculated here to compare
    f1_score(y_val_eva, y_pred,average='macro')


def save_history_plot(model_history, output_graph):
    '''
    gets history of ephos on training a model and saves accury plots
    over the training period in output_graph (should be a string with .png or .pdf extension)
    '''
    fig=plt.figure()
    # summarize history for accuracy
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig(output_graph, bbox_inches='tight')
    
def cross_val(data, labels,model, embedding_matrix,word_index,n,loss_fun, epo): 
    '''
    This function perform a 10-fold cross validation over the data building 10 models on different
    portion of data and averages the accuracy in the final output (takes longer time)
    '''
    from sklearn.model_selection import StratifiedKFold
    X=data
    Y=np.argmax(labels,axis=1)
    seed= 11
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    models=[]
    cvscores = []
    for train, test in kfold.split(X, Y):
        
    
        modelK = build_model(model,embedding_matrix,word_index,n, loss_fun)
        models.append(modelK)
        
        modelK.summary()
        models[-1].fit(X[train], Y[train], epochs=epo, batch_size=128, verbose=1)
    	# evaluate the model
        scores =  models[-1].evaluate(X[test], Y[test], verbose=1)
        print("%s: %.2f%%" % ( models[-1].metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
