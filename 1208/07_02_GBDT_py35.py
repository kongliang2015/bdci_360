# -*- encoding: utf-8 -*-
import sys
import numpy as np
# 随机梯度下降（逻辑斯蒂回归）
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.externals import joblib
from sklearn import metrics

import sys

MODE = sys.argv[1]

##使用分类器对文本向量进行分类训练
def Classifier(infile,clf_file):

    LENGTH = 10000

    data = np.load(infile)
    select = data[:LENGTH,:]

    # 划分数据集
    X = select[:, :-1]
    y = select[:, -1]

    print("shape of X:", X.shape)
    print("shape of y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 10)

    #使用GBDT
    GBDT = GradientBoostingClassifier(n_estimators=40,
                                      learning_rate=0.1,
                                      max_depth=7,
                                      min_samples_split = 700,
                                      min_samples_leaf = 80,
                                      max_features = 40,
                                      subsample = 0.8,
                                      random_state=10)
    GBDT.fit(X_train, y_train)

    print('Test Accuracy: %.2f'%GBDT.score(X_test, y_test))

    joblib.dump(GBDT, clf_file)

    # ROC_curve(GBDT,X_test,y_test)

    # 调整步长(learning rate)和迭代次数(n_estimators)
    # param_test1 = {'n_estimators': range(40, 200, 20)}
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,
    #                                   min_samples_split=700,
    #                                   min_samples_leaf=80, max_depth=7, max_features=40,
    #                                   subsample=0.8, random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    # gsearch1.fit(X, y)

    # 决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split
    # param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
    # gsearch2 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, min_samples_leaf=20,
    #                                          max_features='sqrt', subsample=0.8, random_state=10),
    #     param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(X, y)
    # {'min_samples_split': 500, 'max_depth': 7}
    # 0.553625954472

    # param_test3 = {'min_samples_split': range(300, 1900, 200), 'min_samples_leaf': range(60, 101, 10)}
    # gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, max_depth=7,
    #                                                              max_features='sqrt', subsample=0.8, random_state=10),
    #                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    # gsearch3.fit(X, y)
    # {'min_samples_split': 700, 'min_samples_leaf': 80}
    # 0.551924788459

    # param_test4 = {'max_features': range(10, 200, 10)}
    # gsearch4 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1,
    #                                          n_estimators=40,
    #                                          max_depth=7,
    #                                          min_samples_leaf=80,
    #                                          min_samples_split=700,
    #                                          subsample=0.8, random_state=10),
    #     param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
    # gsearch4.fit(X, y)
    # {'max_features': 40}
    # 0.55135684559

    # param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    # gsearch5 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=40, max_depth=7, min_samples_leaf=80,
    #                                          min_samples_split=700, max_features=40, random_state=10),
    #     param_grid=param_test5, scoring='roc_auc', iid=False, cv=5)
    # gsearch5.fit(X, y)
    # {'subsample': 0.8}
    # 0.55135684559

    # print(gsearch4.grid_scores_)
    # print(gsearch1.best_params_)
    # print(gsearch1.best_score_)

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

    # 读入待预测的数据
    pred_data =  np.load(pred_file)
    test_labels_GBDT = []

    # 读入分类器
    clf = joblib.load(clf_file)

    # 预测标签
    for i in pred_data:
        test_labels_GBDT.append(clf.predict([i]))

    # 读入文本标签
    docid_list = getDocid(docid_file)

    # 编辑预测结果
    pred_dict = dict()

    fw = open(outfile, 'w')

    for idx, lab in enumerate(test_labels_GBDT):
        if lab == 0:
            tag = 'NEGATIVE'
        else:
            tag = 'POSITIVE'

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


def main():
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/model/"

    inFile = dataPath + "train/train_all.npy"

    trainPart = dataPath + "train/train_p1.txt.npy"

    testNpy = dataPath + "test/test_p1.txt.npy"

    testDoc = dataPath + "test/test_p1_docid.txt"

    clf_file = mdlPath + "GBDT_xx.pkl"

    submit = dataPath + "submit_gbdt.csv"

    if MODE == "train":
        # 训练模型
        Classifier(trainPart,clf_file)
    elif MODE == "transform":
        # 预测
        pred(clf_file, testNpy, testDoc, submit)
    else:
        print("mode error")

if __name__ == '__main__':
    main()