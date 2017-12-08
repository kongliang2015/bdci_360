# -*- encoding: utf-8 -*-
from __future__ import print_function
import logging
import numpy as np
from tflearn.data_utils import VocabularyProcessor,to_categorical

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def getLabel(x):
    if x == '__label__NEGATIVE':
        lable = '0'
    elif x== '__label__POSITIVE':
        lable = '1'
    else:
        # print "x=",x
        lable = '0'
    return lable


class MySentences(object):
    def __init__(self, filename,dtype,ptype):
        self.filename = filename
        self.ptype = ptype
        self.dtype = dtype

    def __iter__(self):
        # 读入训练数据
        ctr = 0
        for line in open(self.filename):
            article = line.replace('\n', '').split(" ")
            if self.dtype == 'train':
                # train:读入标签
                title = getLabel(article[0])
                content = article[1:]
            elif self.dtype == 'test':
                # test:读入文档ID
                title = article[0]
                content = article[1:]
            elif self.dtype == 'all':
                content = article[0:]
            # 把list连接成string
            # [['i','like','python'],['hello','world']] -> ['i like python','hello world']
            doc = " ".join(content).decode('utf-8')
            # py27 需要做decode('utf-8') 转为unicode
            # py35 不用
            ctr +=1
            if ctr % 10000 == 0:
                print(str(ctr) + " record has been processed.")
            if self.ptype == 'get_info':
                yield title
            elif self.ptype == 'get_content':
                yield doc

def build_vocabulary(inFile, dtype,vocabFile):
    # 文本长度 200
    MAX_LENGTH = 300
    NB_CLASSES = 2

    # 读入分词后文本
    doc = MySentences(inFile, dtype, 'get_content')
    # 把原始文本映射到index
    processor = VocabularyProcessor(MAX_LENGTH,min_frequency=5)
    processor.fit(doc)
    processor.save(vocabFile)

# 把原始文本转化为由词汇表索引表示的矩阵
def text2npy(inFile,outFile,vocabFile,dtype):

    processor = VocabularyProcessor.restore(vocabFile)
    doc = MySentences(inFile, dtype, 'get_content')
    train_doc = list(processor.transform(doc))

    # 可以使用 to_categorical 来实现onehot编码
    # to_categorical(np.array(lable), NB_CLASSES))

    if dtype == 'train':
        # 把标签做变换
        lable = []
        for y in MySentences(inFile, dtype, 'get_info'):
            lable.append(int(y))
        y = np.array(lable)
        # 保存到文件
        np.save(outFile,np.column_stack([train_doc,y]))
    elif dtype =='test':
        np.save(outFile, train_doc)

        fw = open(outFile+"_doc.txt", 'w')
        for y in MySentences(inFile, dtype, 'get_info'):
            fw.write(y.encode('utf8')+"\n")
        fw.close()

def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"
    
    # 训练数据
    trainText =dataPath + "train/train_all.tsv"
    trainNpy = dataPath + "train/train_all_033.npy"

    # 测试数据
    testText = dataPath + "test/test_all.tsv"
    testNpy = dataPath + "test/test_all_033.npy"

    vocabFile = dataPath + "text_all_vocab"
    text_all =dataPath + "text_all.txt"

    # 创建词表
    # build_vocabulary(text_all, 'all', vocabFile)

    # ####
    trainPartText = [dataPath + "train/train_p" + str(x) + ".txt" for x in range(1,7)]
    trainPartNpy = [dataPath + "train/train_p" + str(x) + ".npy" for x in range(1,7)]
    testPartText = dataPath + "test/test_p1.txt"
    testPartNpy = dataPath + "test/test_p1.npy"

    #
    # for text,npy in zip(trainPartText,trainPartNpy):
    #     text2npy(text,npy,vocabFile,'train')
    # ###
    # 把训练数据转成矩阵
    text2npy(trainText,trainNpy,vocabFile,'train')

    # 把测试数据转成矩阵
    # text2npy(testText, testNpy, vocabFile, 'test')

if __name__ == '__main__':
    main()

