# -*- encoding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


from gensim import models
import logging
import numpy as np
from keras.preprocessing import sequence
import collections

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 获取训练数据
def getDataLen(inFile):
    # 统计所有出现的词
    word_ctr = collections.Counter()
    # 评论的最大长度
    maxlen = 0
    len_ctr = collections.Counter()
    
    # 读入训练数据           
    f=open(inFile)
    lines=f.readlines()
    for line in lines:
        article = line.replace('\n','').split(" ")

        # 内容
        content = article[1:]

        # 获得评论的最大长度
        if len(content) > maxlen:
            maxlen = len(content)

        # 统计各种长度的文章个数
        len_ctr[str(len(content))] += 1


    f.close()
        
    print('max_len ',maxlen)
    print ('len_ctr ', len_ctr)


def getLabel(x):
    if x == '__label__NEGATIVE':
        lable = '0'
    elif x== '__label__POSITIVE':
        lable = '1'
    else:
        print "x=",x
        lable = '0'
    return lable


# In[18]:

# 获取训练数据
def getTrainSet(inFile,ptype):
    # 训练集
    train_set=[]
    title_set = []
    # 读入训练数据  
    f=open(inFile)
    lines=f.readlines()
    for line in lines:
        article = line.replace('\n','').split(" ")
        if ptype == 'train':
            title = getLabel(article[0])
        elif ptype== 'test':
            title = article[0]
        title_set.append(title)
        # 内容
        train_set.append(article[1:])

    f.close()
        
    return (title_set,train_set)

# In[19]:

# 训练word2vec
def trainModel(inFile,modelName):
    # 读入数据    
    _,data_set = getTrainSet(inFile,'train')
    print "train word2vec model use:", inFile

    # 训练
    # 少于min_count次数的单词会被丢弃掉, 默认值为5
    # size = 神经网络的隐藏层的单元数 default value is 100
    # workers= 控制训练的并行:default = 1 worker (no parallelization) 只有在安装了Cython后才有效
    model = models.Word2Vec(data_set,min_count=5,window=10,size = 100,workers=4)

    modelFile = modelName + ".mdl"
    # 存储模型
    model.save(modelFile)

    vecFile = modelName + ".bin"
    # 存储vector
    model.wv.save_word2vec_format(vecFile, binary=True)


# In[20]:

# 把原始文本转化为由词汇表索引表示的矩阵
def fastBuildSeq(inFile,outFile,word_vec,ptype):
    # 读入数据
    title_set,data_set = getTrainSet(inFile,ptype)
    # 使用dir(object)查看对象的属性
    # 对每一个文章做转换      
    # 注意：由于word2vec的向量在训练的时候用的是unicode的编码，
    # 所以在字典里面匹配key的时候，需要把key转化为unicode的编码，使用decode('utf-8')
    transfrom = []
    for news in data_set:
        # trs_news = [word_vec.vocab[w.decode('utf-8')].index for w in news if w.decode('utf-8') in word_vec.vocab]
        # --- 调试
        trs_news = []
        for w in news:
            if w.decode('utf-8') in word_vec.vocab:
                # print "in vocab = ",w.decode('utf-8')
                trs_news.append(word_vec.vocab[w.decode('utf-8')].index)
            else:
                # 对词表里不存在的词，用空格的index来填充
                trs_news.append('34249')
#         # --
        transfrom.append(trs_news)
    
#     for x in transfrom:
#         print x
    
    # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
    # 训练好的w2v词表的index = 0 对应的词汇是空格
    X = sequence.pad_sequences(transfrom,maxlen=300,padding='post')
    
    if ptype == 'train':
        y = np.array([int(i) for i in title_set])
        # 保存到文件
        np.save(outFile,np.column_stack([X,y]))
        print "outFile :",outFile
    elif ptype == 'test':
        np.save(outFile,X)


def incrementTrain(modelName,files):

    modelFile = modelName + ".mdl"
    vecFile = modelName + ".bin"

    # 读入模型
    model = models.Word2Vec.load(modelFile)
    # 读入增量数据
    for f in files:
        _,data_set = getTrainSet(f, 'train')
        print "increment train word2vec model use:", f
        # 增量训练
        model.train(data_set,total_examples=model.corpus_count, epochs=model.iter)
    # 存储模型
    model.save(modelFile)
    # 存储vector
    model.wv.save_word2vec_format(vecFile, binary=True)

def data2Mat(inFile,modelName,partOut,totalOut,ptype):
    
    # 使用训练出的任意一个词向量，把全部train数据转化为向量矩阵
    # 把分词以后的文本转化为供CNN训练的数据矩阵
    # 由于原始数据较大，每10w分割为一个文件，分别转化

    modelFile = modelName + ".mdl"
    # 装载模型
    model = models.Word2Vec.load(modelFile)

    vecFile = modelName + ".bin"
    word_vec = model.wv.load_word2vec_format(vecFile, binary=True)

    for (tf,po) in zip(inFile,partOut):
        fastBuildSeq(tf,po,word_vec,ptype)
    
    # 把转化完成的5个数据矩阵做合并
    mergeNpy(partOut,totalOut)


# In[22]:

def mergeNpy(part,total):
    # 把转化完成的5个数据矩阵做合并
    for idx,f in enumerate(part):
        if idx == 0:
            tmp = np.load(f)
            mat = tmp
        else:
            tmp = np.load(f)
            mat = np.vstack([mat,tmp])
       
    np.save(total,mat)


def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"
    
    # 训练数据
    inFile = dataPath + "train/train_all.tsv"
    modelName = mdlPath + "w2v_v1"
    trainPart = [dataPath + "train/train_p" + str(x) + ".txt" for x in range(1, 6)]
    trainPartMat = [dataPath + "train/train_p"+ str(x) + ".npy" for x in range(1,6)]
    trainTotalMat = dataPath + "train/train_totalMat.npy"
    
    # 测试数据
    testFile = dataPath + "test/test_all.tsv"
    # 分快处理
    testPart = [dataPath + "test/test_p" + str(x) + ".txt" for x in range(1, 6)]
    # 定义输出文件名
    testPartMat = [dataPath + "test/test_p"+ str(x) + ".npy" for x in range(1,6)]
    testTotalMat = dataPath + "test/test_totalMat.npy"
    
    # 训练词向量模型
    # 训练模型使用全部数据
    # trainModel(inFile, modelName)

    # 增量训练
    # 增量训练初期化
    # trainModel(trainPart[0], modelName)
    # incrementTrain(modelName,trainPart[1:])

#     # 把训练数据转成矩阵
    data2Mat(trainPart,modelName,trainPartMat,trainTotalMat,'train')

    # 把测试数据转成矩阵
    # data2Mat(testFile,modelName,testPartMat,testTotalMat,'test')



if __name__ == '__main__':
    main()

