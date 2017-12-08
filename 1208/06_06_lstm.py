# -*- encoding: utf-8 -*-
from __future__ import print_function
from gensim import models
import logging

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Activation,Dense, Embedding, LSTM,Dense, Input, Flatten,Dropout

from keras.models import Sequential,load_model

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

import numpy as np  
import pandas as pd

from sklearn.cross_validation import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_train_set(train_set_x,train_set_y,valid_portion=0.1):
    n_sample=len(train_set_x)
    sidx = np.random.permutation(n_sample)
    n_train = int(np.round(n_sample*(1. - valid_portion)))

    # valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    # valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    # train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    # train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    valid_set_x = train_set_x[sidx[n_train:]]
    valid_set_y = train_set_y[sidx[n_train:]]
    train_set_x = train_set_x[sidx[:n_train]]
    train_set_y = train_set_y[sidx[:n_train]]

    return train_set_x,train_set_y,valid_set_x,valid_set_y

def train_model(data,cnn_mdl):
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
    


    # 构建网络
    HIDDEN_LAYER_SIZE = 128

    # 使用word2vec训练好的嵌入层
    model = Sequential()
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(8))
    model.add(Activation("sigmoid"))
    # ----
    model.add(Dense(2, activation='softmax'))
    # ----
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#     plot_model(model,to_file=cnn_mdl+'.png',show_shapes=True)
    # 优化器我这里用了adadelta，也可以使用其他方法  
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])
    print(model.summary())

    # =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢  
    model.fit(train_set, train_label,epochs=10,batch_size = 100)

    score = model.evaluate(test_set,test_label, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # 存储模型
    saveMdl(model,cnn_mdl+"_f2")


def saveMdl(model,mdlFile):
    # serialize model to JSON
    model_json = model.to_json()
    with open(mdlFile+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(mdlFile+".h5")
    print("Saved "+ mdlFile + " to disk")

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


def loadMdl(mdl):
    # load json and create model
    json_file = open(mdl+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.summary()

    # load weights into new model
    loaded_model.load_weights(mdl+".h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print("Loaded " + mdl + " from disk")

    return loaded_model

def loadPred(text,npy,mdl):

    docid = getTestSet(text)
    print("len of docid list",len(docid))
    # 载入网络模型
    model = loadMdl(mdl + '_left')

    # 载入数据
    data = np.load(npy)
    print('shape of test set:', data.shape)
    lable = model.predict(data)
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
    
def loadTest(npy,mdl):

    # 载入网络模型
    model = loadMdl(mdl + '_left')

    # 载入数据
    data = np.load(npy)

    # 划分数据集
    X = data[:, :-1]
    y = data[:, -1]

    print("shape of X:", X.shape)
    print("shape of y:", y.shape)

    # 把一维的标签做onehot，pd.get_dummies的结果是df,把df转为ndarray (as_matrix())
    train_label = pd.get_dummies(y).as_matrix()

    score = model.evaluate(X,train_label, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

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
    trainPart = dataPath + "train/train_p1.txt.npy"

    textFile = dataPath + "test/test_docid.txt"
    testNpy = dataPath + "test/test_totalMat.npy"


    cnn_mdl = mdlPath + "lstm_v2"
    cnn_part = mdlPath + "lstm_part"
    
    data = np.load(trainPart)
    train_model(data,cnn_part)

    # data = np.load(inFile)
    # trainModel(data,modelFile,cnn_mdl)
    #
    # 测试模型
    # loadTest(inFile[5],cnn_mdl[0])

    # 预测
    # loadPred(textFile,testNpy,cnn_mdl[0])

    #去除可能的重复docid
    # mergeKey(dataPath + "submit_v2.txt")

if __name__ == '__main__':
    main()
