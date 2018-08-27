# -*- coding: utf-8 -*
import os
import sys
import logging
import numpy as np
import keras 
from data_util import load_dict,load_vec
from get_data import get_train_test_tensor,get_batch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from models.model_BiCNN import Bi_CNN
from models.model_BiCNN1 import Bi_CNN1
from models.model_BiCNN2 import Bi_CNN2
from models.model_BiCNN3 import Bi_CNN3
from models.model_CNN import CNN
from models.model_CNN2 import CNN2
from models.model_AttLSTM import AttLSTM
from models.model_BiLSTM import BiLSTM
from models.model_FastText import FastText
from models.model_FastText2 import FastText2
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import heapq
from data_config import Data_tup_v2 




#model name
model_name='FastText'

#parameter
log_path='log/'+model_name+'.log'
model_path='save_models/'+model_name+'.h5'
error_filepath='err_file/'+model_name+'_test_err.csv'
train_error_filepath='err_file/'+model_name+'_train_err.csv'

#log config
logger=logging.getLogger()
logging.basicConfig(level=logging.INFO)
file_handler=logging.FileHandler(log_path,mode='w')
fmt=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)
word_dict=load_dict()
print('loading data ... ... ...')
#load data
train_tensor,train_label,test_tensor,test_label,num_class=get_train_test_tensor()

print('---------------------------------------------')

iter_num=1
maxlen=61
embedding_dim=300
batch_size=128
# num_class=20
print('Build model...')
if model_name=='BiCNN':
    model=Bi_CNN(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='BiCNN1':
    model=Bi_CNN1(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='BiCNN2':
    model=Bi_CNN2(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='BiCNN3':
    model=Bi_CNN3(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='BiCNN1':
    model=CNN(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='AttLSTM':
    model=AttLSTM(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='CNN':
    model=CNN(maxlen,embedding_dim,word_dict,num_class)
# elif model_name=='CNN2':
#     model=CNN2(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='FastText':
    model=FastText(maxlen,embedding_dim,word_dict,num_class)
elif model_name=='FastText2':
    model=FastText2(maxlen,embedding_dim,word_dict,num_class)
print('fit...') 
# train_data=np.array(train_tensor)
# train_label=to_categorical(np.array(train_label),num_classes=num_class)
model.summary(print_fn=logging.info)
#model.fit(train_data,train_label,validation_split=0.1,epochs=10,batch_size=64,callbacks=callback_list,verbose=0)
max_acc=0.0
for iter in range(iter_num):
    error_file=open(error_filepath,'w+')
    i=10
    sum_loss=0.
    sum_acc=0.
    data_length=len(train_tensor)
    batch_nums=(data_length//batch_size)-1
    all_train=[]
    all_train_label=[]
    while i<50:
        batch_data,batch_label=get_batch(batch_size,i,train_tensor,train_label)
        batch_data=np.array(batch_data)
        batch_label=np.array(batch_label)
        batch_label=to_categorical(batch_label,num_classes=num_class)
        batch_train,batch_test,batch_train_label,batch_test_label=train_test_split(batch_data,batch_label,test_size=0.1,random_state=22)

        batch_train=np.array(batch_train)
        batch_test=np.array(batch_test) 
        model.train_on_batch(batch_train, batch_train_label)           
        i+=1
        print('iter : %d, i :%d'%(iter,i))
        loss,acc = model.evaluate(batch_test, batch_test_label,batch_size=batch_size)
        print('Train loss: %f, Train accuracy: %f '%(loss,acc))
        
        sum_loss+=loss
        sum_acc+=acc
    avg_loss=sum_loss/batch_nums
    avg_acc=sum_acc/batch_nums
    print('Epoch %d average loss: %f'%(iter,avg_loss))
    print('Epoch %d average acc: %f'%(iter,avg_acc))


    logging.info('Epoch %d average loss: %f'%(iter,avg_loss))
    logging.info('Epoch %d average acc: %f'%(iter,avg_acc))
    #线下测试集
    test_tensor_np=np.array(test_tensor)
    # test_label_np=np.array(test_label)
    # test_label_np=to_categorical(test_label_np,num_classes=num_class)
    # test_loss,test_acc = model.evaluate([test_tensor_np], test_label_np,batch_size=batch_size)
    predict_res= keras.utils.np_utils.probas_to_classes(model.predict_classes(test_tensor_np))
    print(predict_res)
    #保存模型
 
    logging.info('saving model...')
    model.save(model_path)
