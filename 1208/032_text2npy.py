# -*- encoding: utf-8 -*-
from __future__ import print_function
from gensim import models
import logging
import numpy as np
from keras.preprocessing import text,sequence


# In[ ]:

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:

def getLabel(x):
    if x == '__label__NEGATIVE':
        lable = '0'
    elif x== '__label__POSITIVE':
        lable = '1'
    else:
        # print "x=",x
        lable = '0'
    return lable


# In[ ]:

# 获取训练数据
def getTrainSet(inFile,ptype):
    # 训练集
    # 读入训练数据
    for line in open(inFile):
        article = line.replace('\n','').split(" ")
        if ptype == 'train':
            title = getLabel(article[0])
        elif ptype== 'test':
            title = article[0]
        # 内容
        content = article[1:]
        yield (title,content)


# 把原始文本转化为由词汇表索引表示的矩阵
def text2npy(inFile,outFile,modelFile,vecFile,ptype):

    # 装载模型
    model = models.Word2Vec.load(modelFile)
    word_vec = model.wv.load_word2vec_format(vecFile, binary=True) 
    
    # 使用dir(object)查看对象的属性
    # 对每一个文章做转换      
    # 注意：由于word2vec的向量在训练的时候用的是unicode的编码，
    # 所以在字典里面匹配key的时候，需要把key转化为unicode的编码，使用decode('utf-8')
    transfrom = []
    # 读入数据
    title_list = []

    for title,content in getTrainSet(inFile,ptype):
        trs_news = [word_vec.vocab[w.decode('utf-8')].index for w in content if w.decode('utf-8') in word_vec.vocab]
        transfrom.append(trs_news)
        title_list.append(title)

    # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
    # 训练好的w2v词表的index = 0 对应的词汇是空格
    
    X = sequence.pad_sequences(transfrom,maxlen=300,padding='post')

    # for i in X:
    #     print(i)

    if ptype == 'train':
        y = np.array([int(i) for i in title_list])
        # 保存到文件
        np.save(outFile,np.column_stack([X,y]))
    elif ptype =='test':
        np.save(outFile,X)

# In[ ]:

def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"
    
    # 训练数据
    trainText =dataPath + "train/train_all.tsv"
    modelFile = mdlPath + "w2v_v2.mdl"
    vecFile = mdlPath + "w2v_v2.bin"

    trainNpy = dataPath + "train/train_all.npy"

    # 测试文本+训练文本

    # 测试数据
    testText = dataPath + "test/test_all.tsv"
    testNpy = dataPath + "test/test_all.npy"

    # 把训练数据转成矩阵
    # text2npy(trainText,trainNpy,modelFile,vecFile,'train')

    # 把测试数据转成矩阵
    # text2npy(testText, testNpy, modelFile, vecFile, 'test')

    #
    trainPart = dataPath + "test/test_p1"
    text2npy(trainPart+".txt", trainPart+".npy", modelFile, vecFile, 'test')

if __name__ == '__main__':
    main()

