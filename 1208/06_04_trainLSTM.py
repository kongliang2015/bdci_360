
# coding: utf-8

# In[1]:


import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# In[2]:

import numpy as np

# from keras.layers.core import Activation, Dense
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM

from keras.layers import Activation,Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout

from keras.models import Sequential,load_model

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

import collections  #用来统计词频
import logging
import pandas as pd


# In[3]:

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[4]:

# 获取训练数据
class MySentences(object):
    def __init__(self, filename, dtype, ptype):
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
            ctr += 1
            if ctr % 10000 == 0:
                print(str(ctr) + " record has been processed.")
            if self.ptype == 'get_info':
                yield title
            elif self.ptype == 'get_content':
                yield doc


# 把原始文本转化为由词汇表索引表示的矩阵
def trainLSTM(mtype,inFile,outFile,mdlFile):
    # 读入数据
    # 读入分词后文本
    doc = MySentences(inFile, dtype, 'get_content')
    target_set,data_set,maxlen,word_ctr = getTrainSet(inFile)
    
    # 创建训练数据
    X = np.empty(len(data_set),dtype=list)
    y = np.array([int(i) for i in target_set])
    
#     print maxlen
#     print word_ctr
    
#     print "X len = ",len(X)
#     print "y len = ",len(y)
    
    # ('max_len ', 172)
    # ('nb_words ', 5195)
    MAX_FEATURES = 3800
    MAX_SENTENCE_LENGTH = 100
    
    # 对于不在词汇表里的单词，把它们用伪单词 UNK 代替。 
    # 根据句子的最大长度 (max_lens)，我们可以统一句子的长度，把短句用 0 填充。
    # 接下来建立两个 lookup tables，分别是 word2index 和 index2word，用于单词和数字转换。 
    vocab_size = min(MAX_FEATURES, len(word_ctr)) + 2
    word2index = {x[0]: i+2 for i, x in enumerate(word_ctr.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v:k for k, v in word2index.items()}
    
    np.save("./model/word2index.npy",word2index)
    np.save("./model/index2word.npy",index2word)
    
    # 对每一个文章做转换      

    i = 0
    for news in data_set:
        trs_news = []
        for w in news:
            if w in word2index:
                trs_news.append(word2index[w])
            else:
                trs_news.append(word2index['UNK'])
        X[i] = trs_news
        i += 1
    
    # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
    # 训练好的w2v词表的index = 0 对应的词汇是空格
    X = sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH,padding='post')
    
    np.save(outFile,np.column_stack([X,y]))
    
    # 划分数据
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 把一维的标签做onehot，pd.get_dummies的结果是df,把df转为ndarray (as_matrix())
    ytrain_label = pd.get_dummies(ytrain).as_matrix()
    ytest_label = pd.get_dummies(ytest).as_matrix()
    print('shape of train label:',ytrain_label.shape)  
    print('shape of test label:',ytest_label.shape)
      
   
    # 构建网络
    HIDDEN_LAYER_SIZE = 32
    EMBEDDING_SIZE = 128

    if mtype == 'lstm':
        model = Sequential()
        model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        # ----
        model.add(Dense(3, activation='softmax'))
        # ----
        model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    elif mtype == 'bilstm':
        model = Sequential()
        model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
        model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2)))
#         model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))
        # ----
        model.add(Dense(3, activation='softmax'))
        # ----
        model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
      
    
#     # 参考文章的网络结构
#     model = Sequential()
#     model.add(Embedding(vocab_size, 256, input_length=sentence_max_len))
#     model.add(Bidirectional(LSTM(128,implementation=2)))
#     model.add(Dropout(0.5))
#     model.add(Dense(2, activation='relu'))
#     model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])
    
    # 训练网络
    BATCH_SIZE = 10
    NUM_EPOCHS = 32
    
#     if mtype == 'bilstm':
#         from keras.utils.np_utils import to_categorical
#         ytrain = to_categorical(ytrain)
#         ytest = to_categorical(ytest)
    # lstm
    model.fit(Xtrain, ytrain_label, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest_label))
    
    # bilstm
#     model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))
    
    # 预测 
    score, acc = model.evaluate(Xtest, ytest_label, batch_size=BATCH_SIZE)
#     score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    print('{}   {}      {}'.format('预测','真实','句子'))
    for i in range(10):
        idx = np.random.randint(len(Xtest))
        xtest = Xtest[idx].reshape(1,100)
        ylabel = ytest[idx]
        pred = model.predict(xtest)
        print pred
        ypred = getLabel(pred[0].argmax())
        sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
        print(' {}      {}     {}'.format(ypred, int(ylabel), sent))
    
    # 模型持久化
    model.save(mdlFile) 


def testModel(inFile,testFile,testTxt,mdlFile):
    
    MAX_FEATURES = 3800
    MAX_SENTENCE_LENGTH = 100
    
    # 读入数据
    target_set,data_set,maxlen,word_ctr = getTrainSet(inFile)
    
    # 读入数据
    test_target_set,test_data_set,test_maxlen,test_word_ctr = getTrainSet(testFile)
    
    X = np.empty(len(test_data_set),dtype=list)
    y = np.array([int(i) for i in test_target_set])
    
#     print y
#     y_ = pd.get_dummies(y).as_matrix()
#     print y_
    
    # 读入数据
#     word2index = np.load("./model/word2index.npy")
#     index2word = np.load("./model/index2word.npy")

    vocab_size = min(MAX_FEATURES, len(word_ctr)) + 2
    word2index = {x[0]: i+2 for i, x in enumerate(word_ctr.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v:k for k, v in word2index.items()}
    
    # 对每一个文章做转换      
    i = 0
    for news in test_data_set:
        trs_news = []
        for w in news:
            if w in word2index:
                trs_news.append(word2index[w])
            else:
                trs_news.append(word2index['UNK'])
        X[i] = trs_news
        i += 1
    
    # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
    # 训练好的w2v词表的index = 0 对应的词汇是空格
    X = sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH,padding='post')
    
    # 装载模型
    model = load_model(mdlFile)
    
    testLabel,testText = getTestSet(testTxt)
    
    # 预测 
    print('-------------------------------------------')
    print('{}     |   {}  |    {}'.format('AI評価','人の評価','評価原文'))
    print('-------------------------------------------')
    for i in range(3):
        xtest = X[i].reshape(1,100)
        ylabel = y[i]
        pred = model.predict(xtest)
        # 打印预测出的概率
#         print pred
        ypred = getLabel(pred[0].argmax())
        sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
        print('{}   |  {}   |  {}'.format(getString(ypred), getString(str(ylabel)), testText[i]))
        print('-------------------------------------------')


# In[8]:

def getLabel(x):
    if x == 0:
        lable = '-1'
    elif x== 1:
        lable = '0'
    elif x== 2:
        lable = '1'
    return lable


# In[9]:

def getString(x):
    strlable = ""
    if x == '-1':
        strlable = 'negative'
    elif x== '0':
        strlable = u'neutral'
    elif x== '1':
        strlable = u'positive'
        
    return strlable


# In[10]:

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

    text_all =dataPath + "text_all.txt"
    outFile = dataPath + "text_all_0604.npy"


    lstmMdl = "./model/lstm_mdl.h5"
    bilstmMdl = "./model/bilstm_mdl.h5"
   
    # 把分词以后的文本转化为供LSTM训练的数据文件
#     trainLSTM('lstm',inFile,outFile,lstmMdl)
    trainLSTM('bilstm',text_all,outFile,bilstmMdl)
    
#     testModel(inFile,testFile,testTxt,lstmMdl)


# In[11]:

if __name__ == '__main__':
    main()

