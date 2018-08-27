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
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_3d_block(inputs,TIME_STEPS):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = merge([inputs, a_probs], mode='mul')
    return output_attention_mul

def BiLSTM(maxlen,embedding_dim,word_dict,class_nums):    
   
    print('Build model...')
    input_feature = Input(shape=(maxlen,), dtype='int32', name='input')
    embed = Embedding(len(word_dict),
                        embedding_dim,
                        embeddings_initializer='uniform',
                        input_length=maxlen
                        )(input_feature)

    #LSTM
    lstm_res=LSTM(64,dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(embed)
    lstm_res=LSTM(64,dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(lstm_res) 
    #Attention
    #lstm_res=attention_3d_block(lstm_res,maxlen)
    #lstm_res = GlobalMaxPooling1D()(lstm_res)
    lstm_res = Flatten()(lstm_res)
     
    merge=Dense(64,activation='relu')(lstm_res)
    merge=Dropout(0.3)(merge)
    output=Dense(class_nums,activation='softmax')(merge)

    model = Model(input=[input_feature],output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print(model.summary())

    return model
