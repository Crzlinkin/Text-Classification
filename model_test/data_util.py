# -*- coding: utf-8 -*
import os
import re
import random
from sklearn.model_selection import train_test_split
import numpy as np
# from keras.utils.np_utils import to_categorical
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')

test_file_path='../../../test_corpus/test_0817/test.data'
test_label_path='../../../test_corpus/test_0817/test.labels'
#字典保存路径
dict_path="../util_data/voc.dict"

def load_dict():
    dictionary={}
    dictionary['UNK']=0
    i=1
    f=open(dict_path)
    lines=f.readlines()[1:]
    for line in lines:
        values=line.decode('utf-8').split(",")
        word=values[0]
        dictionary[word]=i
        i+=1
    return dictionary


def filter_sentense(ori_string):
    ori_string = re.sub(u"[+\-——！，/“”‘’\'・:\,♥. ·。？、~@#￥%……&*（）()〈〉<>《》\[\]：【】〔〕]",''.decode("utf-8"),ori_string.decode("utf-8"))
    return ori_string.strip()

#给每一行数据打类标 padding shffle
def get_test_data(maxlen):
    
    test_data_file=open(test_file_path,'r')
    test_label_file=open(test_label_path,'r')

    # class_map=open(class_map_file,'w+')

    test_data_lines=test_data_file.readlines()
    test_label_lines=test_label_file.readlines()

    test_data_arr=[]
    test_label_arr=[]
    for i in range(len(test_data_lines)):
        content=filter_sentense(test_data_lines[i].strip().decode('utf-8'))
        content=content.lower()
        content=content[0:maxlen-1]
        test_data_arr.append(content)
        test_label_arr.append(test_label_lines[i])
    return test_data_arr,test_label_arr
def get_test_tensor(test_data,max_length):
    test_tensor=[]
    word_dict=load_dict()
    for line in test_data:      
        line_tensor=[]
        #print line
        line=line.strip().decode('utf-8')
        #line_arr=line.split(' ')
        for tmp_word in line:
            if tmp_word in word_dict.keys() :
                line_tensor.append(word_dict[tmp_word])
            else:
                line_tensor.append(0)
        line_tensor=padding(line_tensor,max_length)
        test_tensor.append(line_tensor)
    return test_tensor
def padding(ori_data,max_length):
    if len(ori_data)<max_length:
        padding_length=max_length-len(ori_data)
        for i in range(padding_length):
            ori_data.append(0)
    #print(len(line))
    return ori_data
if __name__=='__main__':
    print "data util"

    
