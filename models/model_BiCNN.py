# -*- coding: utf-8 -*
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import LSTM,Flatten
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def Bi_CNN(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        embeddings_initializer='uniform',
                        input_length=maxlen
                        )(input_feature)

    #LSTM
    lstm_res=LSTM(30,dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(embed)
    lstm_res=LSTM(30,dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(lstm_res) 
    lstm_res=Flatten()(lstm_res)
    #Conv
    conv_res1=Conv1D(activation="relu", filters=64, kernel_size=1, strides=1, padding="valid")(embed)
    pool_res1 = GlobalMaxPooling1D()(conv_res1)

    conv_res2=Conv1D(activation="relu", filters=32, kernel_size=3, strides=1, padding="valid")(embed)
    pool_res2 = GlobalMaxPooling1D()(conv_res2)
 
     
    merge_res=concatenate([lstm_res,pool_res1,pool_res2])

    merge=Dropout(0.3)(merge_res)
    merge=Dense(128,activation='relu')(merge)
    merge=Dropout(0.3)(merge)
    merge=Dense(64,activation='relu')(merge)
    merge=Dropout(0.3)(merge)
    output=Dense(class_nums,activation='softmax')(merge)

    model = Model(input=[input_feature],output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
