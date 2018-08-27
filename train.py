# -*- coding: utf-8 -*
import os
import sys
import logging
import numpy as np

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

iter_num=3
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
    while i<batch_nums:
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
    test_label_np=np.array(test_label)
    test_label_np=to_categorical(test_label_np,num_classes=num_class)
    test_loss,test_acc = model.evaluate([test_tensor_np], test_label_np,batch_size=batch_size)
    predict_res=model.predict([test_tensor_np])
    #print(predict_res)
    #保存模型
    if test_acc>max_acc:
        logging.info('saving model...')
        model.save(model_path)
        max_acc=test_acc
    #输出测试集的错误数据
    for i in range(len(test_tensor_np)):
        one_pos=np.argmax(predict_res[i])
        pres=np.zeros_like(predict_res[i])
        pres[one_pos]=1.
        if (test_label_np[i]==pres).all()==False:
            word_dict_reverse={v:k for k,v in word_dict.items()}
            ori_text=''
            for j in test_tensor_np[i]:
                if j!=0:
                    ori_text+=word_dict_reverse[j]
            max_indexs=heapq.nlargest(3, xrange(len(predict_res[i])), predict_res[i].take)
            pre_res=''
            for pro in max_indexs:
                pre_res+=Data_tup_v2[pro][2]+':'+str(predict_res[i][pro])+','
            pre_res='['+pre_res[0:-1]+']'
            #print(pre_res)
            error_file.write(ori_text.encode("utf-8")+','+Data_tup_v2[np.argmax(test_label_np[i])][2]+','+Data_tup_v2[np.argmax(pres)][2]+','+pre_res+u'\n')

    print('train size:',len(train_tensor))
    print('test size:',len(test_tensor))
    print('Test average loss: ',test_loss)
    print('Test average acc: ',test_acc)

    logging.info('train size: %d',len(train_tensor))
    logging.info('test size: %d',len(test_tensor))
    logging.info('Test average loss: %f',test_loss)
    logging.info('Test average acc: %f',test_acc)
    logging.info('------------------------------------------\n')
    error_file.close()
#输出训练数据的判别错误
train_error_file=open(train_error_filepath,'w+')
batch=0
all_train_err=[]
batch_size=2048
batch_nums=(len(train_tensor)//batch_size)-1
while batch<batch_nums:
    print 'evaluate train data %d / %d'%(batch,batch_nums) 
    batch_data,batch_label=get_batch(batch_size,batch,train_tensor,train_label)
    #print(batch_data)
    batch_data=np.array(batch_data)
    batch_label=np.array(batch_label)
    batch_label=to_categorical(batch_label,num_classes=num_class)    
    train_predict = model.predict(batch_data)
    for i in range(len(train_predict)):
        one_pos=np.argmax(train_predict[i])
        pres=np.zeros_like(train_predict[i])
        pres[one_pos]=1.
        if (batch_label[i]==pres).all()==False:
            word_dict_reverse={v:k for k,v in word_dict.items()}
            ori_text=''
            for j in batch_data[i]:
                if j!=0:
                    ori_text+=word_dict_reverse[j]
            max_indexs=heapq.nlargest(3, xrange(len(train_predict[i])), train_predict[i].take)
            pre_res=''
            for pro in max_indexs:
                pre_res+=Data_tup_v2[pro][2]+':'+str(train_predict[i][pro])+','
            pre_res='['+pre_res[0:-1]+']'
            err_str=ori_text.encode("utf-8")+','+Data_tup_v2[np.argmax(batch_label[i])][2]+','+Data_tup_v2[np.argmax(pres)][2]+','+pre_res+u'\n'
            all_train_err.append(err_str)
    batch+=1
print 'writing training error data'
all_train_err=list(set(all_train_err))
for line in all_train_err:
    train_error_file.write(line)
train_error_file.close()