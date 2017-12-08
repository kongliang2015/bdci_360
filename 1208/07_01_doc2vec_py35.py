# -*- encoding: utf-8 -*-
from gensim.models import Doc2Vec,doc2vec
import numpy as np
import logging
import pickle

import sys

MODE = sys.argv[1]


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TaggededDocument = doc2vec.TaggedDocument

SIZE = 100
EPOCHS = 10

def getLabel(x):
    if x == '__label__NEGATIVE':
        lable = '0'
    elif x== '__label__POSITIVE':
        lable = '1'
    else:
        # print "x=",x
        lable = '0'
    return lable

# 获取训练数据
class MySentences(object):
    def __init__(self, filename,dtype,ptype):
        self.filename = filename
        self.ptype = ptype
        self.dtype = dtype

    def __iter__(self):
        # 读入训练数据
        for uid,line in enumerate(open(self.filename)):
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
            doc = " ".join(content)
            # py27 需要做decode('utf-8') 转为unicode
            # py35 不用
            if self.ptype == 'get_info':
                yield title
            elif self.ptype == 'get_content':
                yield TaggededDocument(words=doc.split(" "),tags=['SENT_%s' % uid])


# def prepare_data(train_file,pred_file):
#
#     train_data = MySentences(train_file, 'train', 'get_content')
#
#     pred_data = MySentences(pred_file, 'test', 'get_content')
#
#
#     return (train_data,pred_data)


# 训练doc2vec
def train_model(train_file,pred_file,text_file,model_file):

    # train_data = MySentences(train_file, 'train', 'get_content')

    # pred_data = MySentences(pred_file, 'test', 'get_content')

    text_all = MySentences(text_file, 'all', 'get_content')


    # instantiate our DM and DBOW models
    model_dm = Doc2Vec(min_count=1, window=10, size=SIZE, sample=1e-3, negative=5, workers=4,alpha=0.025, min_alpha=0.025)
    model_dbow = Doc2Vec(min_count=1, window=10, size=SIZE, sample=1e-3, negative=5, dm=0, workers=4,alpha=0.025, min_alpha=0.025)

    # build vocab over all reviews
    # 使用所有的数据建立词典
    doc_all = text_all

    model_dm.build_vocab(doc_all)
    model_dbow.build_vocab(doc_all)

    # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    # for epoch in range(EPOCHS):
    #     model_dm.train(doc_all,total_examples=model_dm.corpus_count,epochs=model_dm.iter)
    #     model_dbow.train(doc_all,total_examples=model_dbow.corpus_count,epochs=model_dbow.iter)

    model_dm.train(doc_all,total_examples=model_dm.corpus_count,epochs=model_dm.iter)
    model_dbow.train(doc_all,total_examples=model_dbow.corpus_count,epochs=model_dbow.iter)

    dm_mdl = model_file + "_dm.mdl"
    dbow_mdl = model_file + "_dbow.mdl"

    dm_bin = model_file + "_dm.bin"
    dbow_bin = model_file + "_dbow.bin"

    # 存储模型
    model_dm.save(dm_mdl)
    model_dbow.save(dbow_mdl)

    # 存储vector
    model_dm.wv.save_word2vec_format(dm_bin, binary=True)
    model_dbow.wv.save_word2vec_format(dbow_bin, binary=True)

#Get training set vectors from our models
def get_vector(dm,dbow, doc, size):

    # vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    # vec = np.array(model[doc.tags[0]]).reshape((1, size))

    dm_vecs = []
    dbow_vecs = []
    # print(dm.docvecs.index_to_doctag[0])
    # print(dm.docvecs[0])
    for item in doc:

        dm_vec = np.array(dm.docvecs[item.tags[0]]).reshape((1, size))

        dbow_vec = np.array(dbow.docvecs[item.tags[0]]).reshape((1, size))

        dm_vecs.append(dm_vec)
        dbow_vecs.append(dbow_vec)

    dm_vecs = np.concatenate(dm_vecs)
    dbow_vecs = np.concatenate(dbow_vecs)

    vecs = np.hstack((dm_vecs, dbow_vecs))

    # print(len(vecs))
    return vecs

def transform_doc2vec(model_file,train_file,test_file):

    # 读入模型
    dm_mdl = model_file + "_dm.mdl"
    dm_bin = model_file + "_dm.bin"
    model_dm = Doc2Vec.load(dm_mdl)
    dm_vec = model_dm.wv.load_word2vec_format(dm_bin, binary=True)

    # 读入模型
    dbow_mdl = model_file + "_dbow.mdl"
    dbow_bin = model_file + "_dbow.bin"
    model_dbow = Doc2Vec.load(dbow_mdl)
    dbow_vec = model_dbow.wv.load_word2vec_format(dbow_bin, binary=True)

    # 读入数据
    train_data = MySentences(train_file, 'train', 'get_content')
    # 把文本转化为向量
    train_vecs = get_vector(model_dm,model_dbow,train_data,SIZE)
    print("train data shape:",train_vecs.shape)

    # 把标签做变换
    lable = []
    for y in MySentences(train_file, 'train', 'get_info'):
        lable.append(int(y))
    y = np.array(lable)
    # 保存到文件
    train_transformed = np.column_stack([train_vecs, y])
    np.save(train_file+".npy", train_transformed)
    print("train transformed shape:", train_transformed.shape)

    # train over test set
    test_data = MySentences(test_file, 'test', 'get_content')
    test_vecs = get_vector(model_dm,model_dbow,test_data,SIZE)
    # 保存到npy
    print("test data shape:", test_vecs.shape)
    np.save(test_file + ".npy", test_vecs)

    fw = open(test_file + "_docid.txt", 'w')
    for y in MySentences(test_file, 'test', 'get_info'):
        fw.write(y + "\n")
    fw.close()

def main():
    # 定义文件路径
    dataPath = "/media/kinux2347/software/DataScience/bdci360_semi/"
    mdlPath = "/media/kinux2347/software/DataScience/bdci360_semi/model/"

    # 训练数据
    trainText =dataPath + "train/train_all.tsv"

    modelFile = mdlPath + "doc2vec_v1"

    # 测试数据
    testText = dataPath + "test/test_all.tsv"

    # 测试数据
    textAll = dataPath + "text_all.txt"

    # 数据准备

    # for i in prepare_data(trainText,testText):
    #     for x,y in i:
    #         print(x[0])

    if MODE == "train":
        # 训练模型
        train_model(trainText, testText,textAll,modelFile)
    elif MODE == "transform":
        # 把文字转成向量
        transform_doc2vec(modelFile, trainText, testText)
    else:
        print("mode error")


if __name__ == '__main__':
    main()


