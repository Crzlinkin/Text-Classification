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
# def focal_loss_fixed(y_true,y_pred):
#     pt_1=tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
#     pt_0=tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
def FastText(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        embeddings_initializer='uniform',
                        #weights=[np.array(word_vec)],
                        input_length=maxlen
                        )(input_feature)

    pool_res = GlobalMaxPooling1D()(embed)

    pool_res=Dropout(0.3)(pool_res)
    output=Dense(class_nums,activation='softmax')(pool_res)

    model = Model(input=[input_feature],output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
