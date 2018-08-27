# -*- coding: utf-8 -*
import os
import sys
import logging
import numpy as np

from data_util import load_dict,get_test_data,get_test_tensor
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
import heapq
from data_config import Data_tup_v2

#parameter
model_name='FastText'
model_path='../save_models/'+model_name+'.h5'
result_filepath='../err_file/'+model_name+'_res.csv'
log_filepath='../log/'+model_name+'_test.log'
#log config
logger=logging.getLogger()
logging.basicConfig(level=logging.INFO)
file_handler=logging.FileHandler(log_filepath,mode='w')
fmt=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)
#加载字典
word_dict=load_dict()
print('loading test data ... ... ...')

#model parameter
maxlen=61
embedding_dim=150
batch_size=128
num_class=19

#load data
test_data,test_label=get_test_data(maxlen)
test_data_tensor=get_test_tensor(test_data,maxlen)
print('---------------------------------------------')

print('Loading model...')

model=load_model(model_path)
print('fit...') 
model.summary(print_fn=logging.info)
#测试集
test_tensor_np=np.array(test_data_tensor)
test_label_np=np.array(test_label)
test_label_np=to_categorical(test_label_np,num_classes=num_class)

test_loss,test_acc = model.evaluate(test_tensor_np, test_label_np,batch_size=batch_size)
predict_res=model.predict(test_tensor_np)
#print(predict_res)
result_file=open(result_filepath,'w+')
word_dict_reverse={v:k for k,v in word_dict.items()}
result_dict={}
test_data_dict={}
for label in test_label:
    #print(Data_tup_v2[int(label.strip())][2])
    if Data_tup_v2[int(label.strip())][2] in test_data_dict.keys():
            test_data_dict[Data_tup_v2[int(label.strip())][2]]=test_data_dict[Data_tup_v2[int(label.strip())][2]]+1
    else:
        test_data_dict[Data_tup_v2[int(label.strip())][2]]=1
print(test_data_dict)
logging.info(test_data_dict)
print(predict_res.shape)
#输出测试集的错误数据
for i in range(len(test_tensor_np)):
    one_pos=np.argmax(predict_res[i])
    pres=np.zeros_like(predict_res[i])
    pres[one_pos]=1.
    
    if (test_label_np[i]==pres).all()==False:
        ori_text=''
        for j in test_tensor_np[i]:
            if j!=0:
                ori_text+=word_dict_reverse[j]
        #predict_res[i]=[float(x) for x in predict_res[i]]
        max_indexs=heapq.nlargest(3, xrange(len(predict_res[i])), predict_res[i].take)

        if Data_tup_v2[np.argmax(test_label_np[i])][2] in result_dict.keys():
            result_dict[Data_tup_v2[np.argmax(test_label_np[i])][2]]=result_dict[Data_tup_v2[np.argmax(test_label_np[i])][2]]+1
        else:
            result_dict[Data_tup_v2[np.argmax(test_label_np[i])][2]]=1
        #result_dict[Data_tup_v2[pro][2]]=result_dict[Data_tup_v2[pro][2]]+1
        #print(pre_res)
        result_file.write(ori_text.encode("utf-8")+','+Data_tup_v2[np.argmax(test_label_np[i])][2]+u'\n')
result_file.close() 
print('test size:',len(test_data_tensor))
print('Test average loss: ',test_loss)
print('Test average acc: ',test_acc)

logging.info('test size: %d',len(test_data_tensor))
logging.info('Test average loss: %f',test_loss)
logging.info('Test average acc: %f',test_acc)
logging.info('------------------------------------------\n')
print(result_dict)
logging.info(result_dict)
print(log_filepath)
