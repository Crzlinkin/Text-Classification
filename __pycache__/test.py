# -*- coding: utf-8 -*
import re
import os
import sys
import heapq
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')
#读取某个文件夹下指定后缀的文件
def getFileName(data_path):
    allfiles = []
    for root,dirs,files in os.walk(data_path):     
        for filespath in files:
            filepath = os.path.join(root, filespath)
            #print(filepath)
            filepath_arr=filepath.split('/')
            allfiles.append(filepath)   
    return allfiles
def fileter_file(data_path,new_floder):
    file_list=getFileName(data_path)
    for file_path in file_list:
        tmp_file=open(file_path,'r+')
        filelines=tmp_file.readlines()
        file_filter=[filter_sentense(''.join(line.decode("utf-8"))) for line in filelines]
        filepath_arr=file_path.split('/')
        filter_file=open(new_floder+filepath_arr[-1],'w')
        filter_file.write('\n'.join(file_filter))
        filter_file.close()
def filter_sentense(ori_string):
    ori_string = re.sub(u"[+\-——！，/“”‘’\'・:\,♥. ·。？、~@#￥%……&*（）()〈〉<>《》\[\]：【】〔〕]",''.decode("utf-8"),ori_string.decode("utf-8"))
    return ori_string.strip()
#"[\.\!\/_,$%^*()+\"\']+-~|[+-——！，“”‘’ ·。？、~@#￥%……&*（）()<>《》：【】〔〕]"


data_path='/home/paopao/Documents/cls-nlp/cls_scene/real_corpus/'
new_floder='/home/paopao/Documents/corpus_filter/'
fileter_file(data_path,new_floder)

# a='苹果345IPHONE铃声《开场》DJremix版'
# print(a.decode("utf-8"))
# print(filter_sentense(a))

# a=np.array([2,5,6,7,8,12,22,90,465,64,0.1,0.8,0.9,0.49,457])
# max_one=0.0
# max_two=0.0
# max_three=0.0
# res=heapq.nlargest(3, xrange(len(a)), a.take)
# print(res)
