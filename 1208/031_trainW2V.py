# -*- encoding: utf-8 -*-
from __future__ import print_function
from gensim import models
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 去除标点符号和特殊符号,停用词
# 获取训练数据

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        # 读入训练数据
        for line in open(self.filename):
            article = line.replace('\n', '').split(" ")
            yield article


# 训练word2vec
def trainModel(inFile,modelName):

    # 读入数据
    sentences = MySentences(inFile)

    # 训练
    # 少于min_count次数的单词会被丢弃掉, 默认值为5
    # size = 神经网络的隐藏层的单元数 default value is 100
    # workers= 控制训练的并行:default = 1 worker (no parallelization) 只有在安装了Cython后才有效
    model = models.Word2Vec(sentences,min_count=10,window=10,size = 200,workers=4)

    modelFile = modelName + ".mdl"
    # 存储模型
    model.save(modelFile)

    vecFile = modelName + ".bin"
    # 存储vector
    model.wv.save_word2vec_format(vecFile, binary=True)


def incrementTrain(modelName,files):

    modelFile = modelName + ".mdl"
    vecFile = modelName + ".bin"

    # 读入模型
    model = models.Word2Vec.load(modelFile)
    # 读入增量数据
    for f in files:
        data_set = MySentences(f)
        print("increment train word2vec model use:"+f)
        # 增量训练
        model.train(data_set,total_examples=model.corpus_count, epochs=model.iter)
    # 存储模型
    model.save(modelFile)
    # 存储vector
    model.wv.save_word2vec_format(vecFile, binary=True)

def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"
    
    # 训练数据
    inFile = dataPath + "train/train_all.tsv"
    modelName = mdlPath + "w2v_v2"

    
    # 训练词向量模型
    # 训练模型使用全部数据
    trainModel(inFile, modelName)

    # 增量训练
    # 增量训练初期化
    # trainModel(trainPart[0], modelName)
    # incrementTrain(modelName,trainPart[1:])

if __name__ == '__main__':
    main()

