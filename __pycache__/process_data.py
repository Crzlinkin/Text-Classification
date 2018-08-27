# -*- coding: utf-8 -*
import os
import re
import random
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils.np_utils import to_categorical
data_path='new_data/data/'
train_data_path="generate_data/train.csv"
test_data_path="generate_data/test.csv"
dict_path="voc.dict"
#读取某个文件夹下指定后缀的文件
def getFileName():
    allfiles = []
    for root,dirs,files in os.walk(data_path):     
        for filespath in files:
            filepath = os.path.join(root, filespath)
            #print(filepath)
            filepath_arr=filepath.split('/')
            #filename=filepath_arr[-1]
            #print(filename)
            allfiles.append(filepath)
#             filename_arr=filename.split('.')
#             if len(filename_arr)>1:
#                 if filename_arr[-2]=='na' and filename_arr[-1]=='data':
#                     allfiles.append(filepath)
            
    return allfiles

#统计每个文件的行数
def countFileLine(filepath):
    sum_line=0
    tmp_file=open(filepath,'r')
    filelines=tmp_file.readlines()
    sum_line=len(filelines)
    return sum_line

#获取数据字典文件
def get_data_dict():
    file_list=getFileName()
    all_data_dict={}
    #写入字典文件
    dict_file=open(dict_path,'w+')
    for file_path in file_list:
        tmp_file=open(file_path,'r')
        filelines=tmp_file.readlines()
        for line in filelines:
            content=line.split(',')[0]
            content=filter_sentense(content.lower())
            content_arr=content.split(' ')
            for word in content_arr:
                word=word.strip()
                if word in all_data_dict.keys():
                    all_data_dict[word]=all_data_dict[word]+1
                else:
                    all_data_dict[word]=1
        tmp_file.close()
    all_data_dict_up={}
    for ele in all_data_dict:
        if all_data_dict[ele]>10:
            all_data_dict_up[ele]=all_data_dict[ele]
    all_data_dict_up=sorted(all_data_dict_up.items(),key=lambda x:x[1],reverse=True)
    print(all_data_dict_up)
    dict_file.write('UNK\n')
    for word,count in all_data_dict_up:
        res=word+','+str(count)+'\n'
        dict_file.write(res)
    dict_file.close()
    return all_data_dict_up
            
def load_dict():
    dictionary={}
    dictionary['UNK']=0
    i=1
    f=open(dict_path)
    lines=f.readlines()[1:]
    for line in lines:
        values=line.split(",")
        word=values[0]
        dictionary[word]=i
        i+=1
    return dictionary


#给每一行数据打类标 padding shffle
def get_all_data():
    file_list=getFileName()
    train_file=open(train_data_path,'w+')
    test_file=open(test_data_path,'w+')
    all_data=[]
    all_label=[]
    i=0
    for file_path in file_list:
        tmp_file=open(file_path,'r')
        filelines=tmp_file.readlines()
        for line in filelines:
            content=line.split(',')[0]
            content=filter_sentense(content.lower())
            content_arr=content.split(' ')
            if len(content_arr)<500:
                all_data.append(content)
                all_label.append(i)
        i+=1
    #shuffle data
    shuffle_data=list(zip(all_data,all_label))
    random.shuffle(shuffle_data)
    all_data,all_label=zip(*shuffle_data)
    train_data,test_data,train_label,test_label=train_test_split(all_data,all_label,test_size=0.1,random_state=22)
    #train_data,dev_data,train_label,dev_label=train_test_split(train_data,train_label,test_size=0.3,random_state=22)
    #print(all_data)
    #print(all_label)
    for i in range(len(train_data)):
        train_file.write(train_data[i]+','+str(train_label[i])+'\n')
    for i in range(len(test_data)):
        test_file.write(test_data[i]+','+str(test_label[i])+'\n')
    train_file.close()
    test_file.close()
    return train_data,train_label,test_data,test_label

def write_file(data,label,path):
    #print(data)
    #print(label)
    res_file=open(path,'w+')
    for i in range(len(data)):
        res_line=data[i]+','+str(label[i])+'\n'
        res_file.write(res_line)
    res_file.close()

def filter_sentense(ori_string):
    ori_string=ori_string.replace('?',' ')
    ori_string=ori_string.replace('!',' ')
    ori_string=ori_string.replace('¿',' ')
    ori_string=ori_string.replace('¡',' ')
    ori_string=ori_string.replace(',',' ')
    ori_string=ori_string.replace('.',' ')
    ori_string=ori_string.replace('"',' ')
    ori_string=ori_string.replace(':',' ')
    ori_string=ori_string.replace('&amp;',' ')
    ori_string=ori_string.replace('(',' ')
    ori_string=ori_string.replace(')',' ')
    ori_string=ori_string.replace('*',' ')
    ori_string=ori_string.replace('/',' ')
    ori_string=ori_string.replace(';',' ')
    ori_string=ori_string.replace('e-mail','email')
    ori_string=ori_string.replace('-',' ')
    ori_string=ori_string.replace('’',' ')
    ori_string=ori_string.replace('´',' ')
    ori_string=ori_string.replace('\\',' ')
    ori_string = re.sub("[0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ",ori_string)  
    return ori_string.strip()


if __name__=='__main__':
    #file_path_list=getFileName()
    # data_dict=get_data_dict()
    # print(data_dict)
    # train_data,train_label,test_data,test_label=get_all_data()

    # print(len(train_data))
    # print(len(train_label))
    # print(len(test_data))
    # print(test_label)
    # test_label=np.array(test_label)
    # test_label=to_categorical(test_label,num_classes=20)
    # print(test_label)
    res=load_dict()
    print(res['scotts'])
