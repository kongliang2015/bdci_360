# -*- encoding: utf-8 -*-
import sys
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.externals import joblib
from sklearn import metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd

def prepare_data(infile):
    LENGTH = 100000

    data = np.load(infile)
    select = data[:LENGTH, :]

    # 划分数据集
    X = select[:, :-1]
    y = select[:, -1]

    print("shape of X:", X.shape)
    print("shape of y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    return (X_train, X_test, y_train, y_test)

def modelfit(alg, X_train,y_train, X_test,y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(X_test)
    dtrain_predprob = alg.predict_proba(X_test)[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y_test, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, dtrain_predprob)

##使用分类器对文本向量进行分类训练
def xgbClf(prepare_data,clf_file):

    X_train, X_test, y_train, y_test = prepare_data

    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        seed=27)

    modelfit(xgb1, X_train, X_test, y_train, y_test)


def xgbtrain(prepare_data,clf_file,pred_file,docid_file,outfile):

    X_train, X_test, y_train, y_test = prepare_data

    # xgb矩阵赋值
    xgb_val = xgb.DMatrix(X_test, label=y_test)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    #
    params = {
        # 通用参数
        'booster': 'gbtree',
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'nthread': 4,  # cpu 线程数
        # 学习目标参数
        'objective': 'binary:logistic',  # 2分类的问题
        # 'num_class': 2,  # 类别数，与 multisoftmax 并用
        'eval_metric': 'auc',
        'seed': 0,
        # booster参数
        'gamma': 0,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 6,  # 构建树的深度，越大越容易过拟合
        'lambda': 1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 1,  # 随机采样训练样本
        'colsample_bytree': 0.8,  # 生成树时进行的列采样
        'min_child_weight': 1,
        'scale_pos_weight' : 1,
        'eta': 0.3  # 如同学习率
    }

    plst = list(params.items())
    num_rounds = 35  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

    #使用xgboost
    #
    # 首先调整max_depth ,通常max_depth 这个参数与其他参数关系不大，初始值设置为10，
    #       找到一个最好的误差值，然后就可以调整参数与这个误差值进行对比。比如调整到8，
    #       如果此时最好的误差变高了，那么下次就调整到12；如果调整到12,误差值比10 的低，那么下次可以尝试调整到15.
    # 在找到了最优的max_depth之后，可以开始调整subsample,初始值设置为1，然后调整到0.8
    #       如果误差值变高，下次就调整到0.9，如果还是变高，就保持为1.0
    # 接着开始调整min_child_weight , 方法与上面同理
    # 再接着调整colsample_bytree
    # 经过上面的调整，已经得到了一组参数，这时调整eta 到0.05，
    #       然后让程序运行来得到一个最佳的num_round,(在 误差值开始上升趋势的时候为最佳 )

    model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=50)
    model.save_model(clf_file)  # 用于存储训练出的模型

    # 计算错误率
    y_hat = model.predict(xgb_val)
    y = xgb_val.get_label()
    print 'y_hat'
    print y_hat
    print 'y'
    print y
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print '样本总数：\t', len(y_hat)
    print '错误数目：\t%4d' % error
    print '错误率：\t%.5f%%' % (100 * error_rate)



def getDocid(inFile):

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

def pred(clf_file,pred_file,docid_file,outfile):

    bst = xgb.Booster(model_file=clf_file)
    # 读入待预测的数据
    pred_data = np.load(pred_file)

    xgb_test = xgb.DMatrix(pred_data)

    pred = bst.predict(xgb_test)

    # 读入文本标签
    docid_list = getDocid(docid_file)

    # 编辑预测结果
    pred_dict = dict()

    fw = open(outfile, 'w')


    for idx, lab in enumerate(pred):

        if lab < 0.5:
            tag = 'NEGATIVE'
        else:
            tag = 'POSITIVE'
        if docid_list[idx] == "":
            print("null=",idx)
        pred_dict[docid_list[idx]] = tag

    for k, v in pred_dict.items():
        fw.write(k + "," + v + "\n")

    fw.close()

##绘出ROC曲线，并计算AUC
def ROC_curve(lr,x_test,y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(x_test)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()

def mergeKey(infile):

    pred_dict = dict()
    check_dict = dict()
    # 读入训练数据
    fw = open(infile + "_merge.csv", 'w')

    f = open(infile)
    lines = f.readlines()
    ctr = 0
    for line in lines:
        ctr += 1
        article = line.replace('\n', '').split(',')
        pred_dict[article[0]]=article[1]
        if article[0] not in check_dict.keys():
            check_dict[article[0]] = 1
        else:
            check_dict[article[0]] += 1
        print(ctr,":line")

    for k,v in pred_dict.items():
        fw.write(k + ","+ v + '\n')

    print("merge end")
    for k, v in check_dict.items():
        if v >1:
            print(k + "," + v + '\n')

    f.close()
    fw.close()


def main():
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"

    inFile = dataPath + "train/train_all.npy"

    trainPart = dataPath + "train/train_p5.txt.npy"

    testNpy = dataPath + "test/test_all.tsv.npy"

    testDoc = dataPath + "test/test_all.tsv_docid.txt"

    clf_file = mdlPath + "xgboost_xx.pkl"

    submit = dataPath + "submit_xgb_p5.csv"

    # 训练模型
    # xgbtrain(prepare_data(trainPart),clf_file,testNpy, testDoc, submit)

    # 预测
    pred(clf_file, testNpy, testDoc, submit)

    # 去除可能的重复docid
    mergeKey(submit)


if __name__ == '__main__':
    main()