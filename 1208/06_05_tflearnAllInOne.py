# -*- encoding: utf-8 -*-
from __future__ import print_function
from tflearn.data_utils import VocabularyProcessor,to_categorical
import logging

import numpy as np  
import pandas as pd

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout,fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_train_set(train_set_x,train_set_y,valid_portion=0.1):
    n_sample=len(train_set_x)
    sidx = np.random.permutation(n_sample)
    n_train = int(np.round(n_sample*(1. - valid_portion)))

    valid_set_x = train_set_x[sidx[n_train:]]
    valid_set_y = train_set_y[sidx[n_train:]]
    train_set_x = train_set_x[sidx[:n_train]]
    train_set_y = train_set_y[sidx[:n_train]]
    return train_set_x,train_set_y,valid_set_x,valid_set_y

def conv_model(network):
    branch1 = conv_1d(network,200,3,padding='valid',activation='relu',regularizer="L2")
    branch2 = conv_1d(network, 200, 4, padding='valid', activation='relu',regularizer="L2")
    branch3 = conv_1d(network, 200, 5, padding='valid', activation='relu',regularizer="L2")
    network = merge([branch1,branch2,branch3],mode='concat',axis=1)
    network = tf.expand_dims(network,2)
    network = global_max_pool(network)
    network = dropout(network,0.5)

    return network

def build_model(vocabFile,model_type = 'bilstm'):

    processor = VocabularyProcessor.restore(vocabFile)
    n_words = len(processor.vocabulary_)

    net = tflearn.input_data([None,300])
    net = tflearn.embedding(net,input_dim=n_words,output_dim=200)

    if model_type == 'bilstm':
        net = tflearn.bidirectional_rnn(net,tflearn.BasicLSTMCell(200),tflearn.BasicLSTMCell(200))
        net = dropout(net, 0.5)
    elif model_type == 'lstm':
        net = tflearn.lstm(net,200,dropout=0.5)
        net = dropout(net, 0.5)
    elif model_type == 'cnn':
        net = conv_model(net)

    net = tflearn.fully_connected(net,2,activation='softmax')
    net = tflearn.regression(net,optimizer='adam',learning_rate=0.05,loss='categorical_crossentropy')

    return net

def train_model(data,vocabFile,cnn_mdl,mode_type):
    # 划分数据集
    X =data[:,:-1]
    y =data[:,-1]
    
    print("shape of X:",X.shape)
    print("shape of y:",y.shape)
    
    train_set,train_tag,test_set,test_tag = split_train_set(X,y,valid_portion = 0.1)
    
    # 把一维的标签做onehot，pd.get_dummies的结果是df,把df转为ndarray (as_matrix())
    train_label = pd.get_dummies(train_tag).as_matrix()
    test_label = pd.get_dummies(test_tag).as_matrix()
    
    print('shape of train set:',train_set.shape)
    print('shape of train label:',train_label.shape)
    print('shape of test set:',test_set.shape)
    print('shape of test label:',test_label.shape)

    net = build_model(vocabFile,mode_type)

    model = tflearn.DNN(net, tensorboard_verbose=0)

    model.fit(train_set,train_label,validation_set=(test_set,test_label),show_metric=False,batch_size=128,n_epoch=16)

    model.save(cnn_mdl+ mode_type)

def getTestSet(inFile):

    # 标签集
    docid_list = []

    # 读入训练数据
    f=open(inFile)
    lines=f.readlines()
    for line in lines:
        article = line.replace('\n','')

        docid_list.append(article)

    f.close()
    return docid_list


def loadPred(text,npy,model_save,vocabFile,mode_type):

    # 载入网络模型
    net = build_model(vocabFile,mode_type)
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load(model_save+mode_type)

    # 载入数据
    data = np.load(npy)
    print('shape of test set:', data.shape)

    docid = getTestSet(text)
    # 预测
    lable = model.predict(data)
    # 预测结果文件
    output = text + "_submit.txt"

    pred_dict = dict()

    fw = open(output, 'w')

    for idx,lab in enumerate(lable):
        if lab[0]>lab[1]:
            tag = 'NEGATIVE'
        else:
            tag = 'POSITIVE'

        pred_dict[docid[idx]] = tag

    for k,v in pred_dict.items():
        fw.write(k + ","+ v + "\n")

    fw.close()
#
def mergeKey(infile):

    pred_dict = dict()
    # 读入训练数据
    fw = open(infile + "_merge.txt", 'w')

    f = open(infile)
    lines = f.readlines()
    for line in lines:
        article = line.replace('\n', '').split(',')
        pred_dict[article[0]]=article[1]

    for k,v in pred_dict.items():
        fw.write(k + ","+ v + "\n")

    f.close()
    fw.close()

def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"
    
    inFile = dataPath + "train/train_all.npy"
    trainPart = dataPath + "train/train_p1.npy"

    testNpy = dataPath + "test/test_p1.npy"
    testDoc = testNpy + "_doc.txt"

    cnn_mdl = mdlPath + "cnn_v2"
    lstm_mdl = mdlPath + "lstm_v2"
    cnn_part = mdlPath + "cnn_part"
    vocabFile = dataPath + "text_all_vocab"
    
    # data = np.load(trainPart)
    # train_model(data,vocabFile,lstm_mdl,'lstm')

    # 预测
    # loadPred(testDoc,testNpy,cnn_part,vocabFile,'bilstm')

    #去除可能的重复docid
    mergeKey("/home/hadoop/submit_gbdt_p4.csv")

if __name__ == '__main__':
    main()
