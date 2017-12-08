# -*- encoding: utf-8 -*-
from __future__ import print_function
from gensim import models
import logging

from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding  
from keras.models import Sequential,load_model
from keras.layers import Merge 
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def trainModel(data,modelFile,cnn_mdl):
    # 划分数据集
    X =data[:,:-1]
    y =data[:,-1]
    
    print("shape of X:",X.shape)
    print("shape of y:",y.shape)
    
    train_set,test_set,train_tag,test_tag = train_test_split(X,y,test_size = 0.3)
    
    # 把一维的标签做onehot，pd.get_dummies的结果是df,把df转为ndarray (as_matrix())
    train_label = pd.get_dummies(train_tag).as_matrix()
    test_label = pd.get_dummies(test_tag).as_matrix()
    
    print('shape of train set:',train_set.shape)
    print('shape of train label:',train_label.shape)
    print('shape of test set:',test_set.shape)
    print('shape of test label:',test_label.shape)
    
    # 读入模型   
    model = models.Word2Vec.load(modelFile)
    # 读入word2vec模型提供的嵌入层，权重需要训练
    model_embedding = model.wv.get_embedding_layer(train_embeddings=False)

    # 训练嵌入层权重
    kmodel = Sequential()
    kmodel.add(model_embedding)
    kmodel.compile('rmsprop', 'mse')
    
#     print(model_embedding.get_weights())
    
    # 定义嵌入层
    # trainable=True 通过训练来更新权重
    # trainable=Fasle 由于使用了word2vec提供的权重，这里不用再训练了
    embedding_layer = Embedding(input_dim = model_embedding.input_dim,
                                output_dim = 200,
                                input_length = 300,
                                weights = model_embedding.get_weights(),
                                trainable = False)
    
    model_left = Sequential() 
    
    # 使用word2vec训练好的嵌入层
    model_left.add(embedding_layer)
    
    model_left.add(Conv1D(300, 3, padding='same',activation='relu'))
    model_left.add(MaxPooling1D(3)) 
    model_left.add(Conv1D(300, 3, padding='same',activation='relu'))
    model_left.add(MaxPooling1D(3)) 
    model_left.add(Conv1D(300, 3, padding='same',activation='relu'))
    model_left.add(MaxPooling1D(18))
    model_left.add(Flatten())
 
    model_right = Sequential() 
    
    # 使用word2vec训练好的嵌入层
    model_right.add(embedding_layer)
    
    model_right.add(Conv1D(300, 4, padding='same',activation='relu'))
    model_right.add(MaxPooling1D(4)) 
    model_right.add(Conv1D(300, 4, padding='same',activation='relu'))
    model_right.add(MaxPooling1D(4)) 
    model_right.add(Conv1D(300, 4, padding='same',activation='relu'))
    model_right.add(MaxPooling1D(12))
    model_right.add(Flatten()) 
    
    model_cent = Sequential() 
    
    # 使用word2vec训练好的嵌入层
    model_cent.add(embedding_layer)
    
    model_cent.add(Conv1D(300, 5, padding='same',activation='relu'))
    model_cent.add(MaxPooling1D(5)) 
    model_cent.add(Conv1D(300, 5, padding='same',activation='relu'))
    model_cent.add(MaxPooling1D(5)) 
    model_cent.add(Conv1D(300, 5, padding='same',activation='relu'))
    model_cent.add(MaxPooling1D(8))
    model_cent.add(Flatten())

    merged = Merge([model_left, model_right,model_cent], mode='concat')
    
    # 最终                       
    model = Sequential() 
                   
    model.add(merged)
    model.add(Dense(128, activation='relu')) # 全连接层  
    model.add(Dense(2, activation='softmax')) # softmax，输出文本属于20种类别中每个类别的概率  
        
    # 优化器我这里用了adadelta，也可以使用其他方法  
    model.compile(loss='categorical_crossentropy',  
                  optimizer='Adadelta',  
                  metrics=['accuracy'])  

    # =下面开始训练，nb_epoch是迭代次数，可以高一些，训练效果会更好，但是训练会变慢  
    model.fit(train_set, train_label,epochs=10,batch_size = 100)  

    # score = model.evaluate(train_set, train_label, verbose=0) # 评估模型在训练集中的效果，准确率约99%
    # print('train score:', score[0])
    # print('train accuracy:', score[1])
    
    # score = model.evaluate(test_set,test_label, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    
    # model.save(cnn_mdl)
    dataPath = "/media/kinux2347/software/DataScience/bdci360_semi/"
    testNpy = dataPath + "test/test_all.npy"
    testText = dataPath + "test/test_all.tsv"

    docid = getTestSet(testText)
    print("len of docid list", len(docid))

    # 载入网络模型

    # 载入数据
    data = np.load(testNpy)
    print('shape of test set:', data.shape)
    lable = model.predict(data)
    output = dataPath + "submit_v3.txt"

    pred_dict = dict()

    fw = open(output, 'w')

    for idx, lab in enumerate(lable):
        if lab[0] > lab[1]:
            tag = 'NEGATIVE'
        else:
            tag = 'POSITIVE'

        pred_dict[docid[idx]] = tag

    for k, v in pred_dict.items():
        fw.write(k + "," + v + "\n")

    fw.close()

def getTestSet(inFile):
    # 训练集
    docid_set = []

    # 读入训练数据
    f = open(inFile)
    lines = f.readlines()
    for line in lines:
        article = line.replace('\n', '').split(" ")

        # 文章id
        docid_set.append(article[0])


    f.close()
    return docid_set

def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"
    
    inFile = dataPath + "train/train_all.npy"

    testFile = dataPath + "test/test_all.npy"
    testText = dataPath + "test/test_all.tsv"
    
    modelFile = mdlPath + "w2v_v2.mdl"
    
    cnn_mdl = mdlPath + "cnn_mdl.h5"

    outfile = dataPath + "test/pred.txt"
    
    data_all = np.load(inFile)

    testData = np.load(testFile)

    # 训练模 
    trainModel(data_all,modelFile,cnn_mdl)
    
    # 测试模型
    # pred(testText,testData,modelFile,outfile)


# In[ ]:

if __name__ == '__main__':
    main()

