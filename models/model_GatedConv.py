# -*- coding: utf-8 -*
import numpy as np
from keras.models import *
from keras.layers import *
import tensorflow as tf
import keras as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import LSTM,Flatten
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
def conv_op(fan_in, shape, name):
    W = tf.get_variable("%s_W"%name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
    b = tf.get_variable("%s_b"%name, shape[-1], tf.float32, tf.constant_initializer(1.0))
    return tf.add(tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME'), b)
def CNN(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        embeddings_initializer='uniform',
                        input_length=maxlen
                        )(input_feature)

    
    #Conv
    conv1_res=Conv1D(activation="relu", filters=60, kernel_size=1, strides=1, padding="valid")(embed)
    pool1_res = GlobalMaxPooling1D()(conv1_res)

    conv2_res=Conv1D(activation="relu", filters=30, kernel_size=2, strides=1, padding="valid")(embed)
    pool2_res = GlobalMaxPooling1D()(conv2_res)

    conv3_res=Conv1D(activation="relu", filters=20, kernel_size=3, strides=1, padding="valid")(embed)
    pool3_res = GlobalMaxPooling1D()(conv3_res)

    conv4_res=Conv1D(activation="relu", filters=20, kernel_size=4, strides=1, padding="valid")(embed)
    pool4_res = GlobalMaxPooling1D()(conv4_res)

    merge_res=concatenate([pool1_res,pool2_res,pool3_res,pool4_res])
    
    # merge=Dropout(0.3)(merge_res)
    # merge=Dense(128,activation='relu')(merge)
    merge=Dropout(0.3)(merge_res)
    merge=Dense(64,activation='relu')(merge)
    merge=Dropout(0.3)(merge)
    output=Dense(class_nums,activation='sigmoid')(merge)

    model = Model(input=[input_feature],output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
