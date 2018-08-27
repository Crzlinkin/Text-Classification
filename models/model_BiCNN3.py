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
def LeakReLU(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        f1=0.5*(1+leak)
        f2=0.5*(1-leak)
        res=f1*x+f2*abs(x)
        res=tf.clip_by_value(res,0.,6.)
        return res
def Bi_CNN3(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        #weights=[np.array(word_vec)],
                        embeddings_initializer='uniform',
                        input_length=maxlen
                        )(input_feature)
    embed_fla=Flatten()(embed)
    embed_des=Dense(128,activation='relu')(embed_fla)
    #LSTM
    lstm_res=LSTM(64,dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(embed)
    lstm_fla=Flatten()(lstm_res)
    lstm_des=Dense(128,activation='relu')(lstm_fla)
    #lstm_res = GlobalMaxPooling1D()(lstm_res)
  
    #Conv
    conv_res=Conv1D(activation="relu", filters=64, kernel_size=3, strides=1, padding="valid")(embed)
    conv_fla=Flatten()(conv_res)
    conv_des=Dense(128,activation='relu')(conv_fla)
    conv_des=Lambda(lambda x:LeakReLU(x))(conv_des)
    #pool_res= GlobalMaxPooling1D()(conv_res)

    
    #embed_res = GlobalMaxPooling1D()(embed)
 
     
    merge_res=concatenate([embed_des,lstm_des,conv_des])
    merge_res=Dense(64,activation='relu')(merge_res)
    merge_res=Lambda(lambda x:LeakReLU(x))(merge_res)
    merge=Dropout(0.3)(merge_res)
    output=Dense(class_nums,activation='softmax')(merge)

    model = Model(input=[input_feature],output=[output])
    #n_adam=optimizers.nadam(lr=0.001,beta_1=0.8,beta_2=0.99)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
