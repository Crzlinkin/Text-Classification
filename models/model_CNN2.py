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

def CNN2(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        embeddings_initializer='uniform',
                        input_length=maxlen
                        )(input_feature)
    print(embed)
    embed=Reshape((maxlen,embedding_dim,1))(embed)
    print(embed)
    #Conv
    conv_res=Conv2D(activation="relu", filters=64, kernel_size=(10,10), strides=1, padding="same")(embed)
    pool_res = MaxPooling2D(pool_size=(10,10),strides=1,padding='valid')(conv_res)
    conv_res=Conv2D(activation="relu", filters=32, kernel_size=(10,10), strides=1, padding="same")(pool_res)
    pool_res = MaxPooling2D(pool_size=(10,10),strides=1,padding='valid')(conv_res)
    # conv_res=Conv2D(activation="relu", filters=16, kernel_size=(10,10), strides=1, padding="same")(pool_res)
    # pool_res = MaxPooling2D(pool_size=(3,3),strides=1,padding='valid')(conv_res)
    fla_pool=Flatten()(pool_res)
    # merge=Dropout(0.3)(merge_res)
    # merge=Dense(128,activation='relu')(merge)
    merge=Dropout(0.3)(fla_pool)
    merge=Dense(64,activation='relu')(merge)
    merge=Dropout(0.3)(merge)
    output=Dense(class_nums,activation='softmax')(merge)

    model = Model(input=[input_feature],output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
