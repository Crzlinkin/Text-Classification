# -*- coding: utf-8 -*
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import LSTM,Flatten
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def FastText2(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        embeddings_initializer='uniform',
                        input_length=maxlen
                        )(input_feature)
    pool_embed = GlobalMaxPooling1D()(embed)

    conv_res1=Conv1D(activation="relu", filters=64, kernel_size=3, strides=1, padding="valid")(embed)
    pool_conv1 = GlobalMaxPooling1D()(conv_res1)

    conv_res2=Conv1D(activation="relu", filters=64, kernel_size=2, strides=1, padding="valid")(embed)
    pool_conv2 = GlobalMaxPooling1D()(conv_res2)

    conv_res3=Conv1D(activation="relu", filters=64, kernel_size=1, strides=1, padding="valid")(embed)
    pool_conv3 = GlobalMaxPooling1D()(conv_res3)


    merge_res=concatenate([pool_embed,pool_conv1,pool_conv2,pool_conv3])
    #pool_res=Dropout(0.3)(merge_res)
    output=Dense(class_nums,activation='softmax')(merge_res)

    model = Model(input=[input_feature],output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
