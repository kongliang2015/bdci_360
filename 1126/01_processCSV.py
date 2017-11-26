
import sys
import csv
import jieba
import re
import MeCab


# In[ ]:

# reload(sys)
# sys.setdefaultencoding('utf-8')
csv.field_size_limit(sys.maxsize)


# 过滤文章里的标点和特殊符号 写入到文件
def doFilter(infile,ptype):

    if ptype == 'train':
        # 训练数据4个字段
        COND = 4
    elif ptype == 'test':
        # 测试数据3个字段
        COND = 3

    with open(infile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # 读取内容
        # 用csv.reader 读取 line 已经是list了
        for line in reader:
            if len(line) == COND:
                # 标题过滤
                seg_title = re.sub("\w","",line[1].strip())
                seg_title_return = " ".join(seg_title)
                #                 print ",".join(seg_title)
                # 正文过滤
                seg_content = re.sub("\w","",line[2].strip())
                seg_content_return = " ".join(seg_content)

                if len(seg_title_return) > 0 or len(seg_content_return) > 0:
                    yield seg_title_return + seg_content_return


def makeFilterFile(inputFiles, outfile_all, ptype):
    # 定义训练文件

    # 每次获取最大件数
    STOP_SIZE = 1000001
    # 初始化ctr值
    ctr = 0
    fw = open(outfile_all, 'w')
    marker = {}
    for idx,file in enumerate(inputFiles):
        print("*** <" + ptype[idx] + ">数据过滤处理开始 ***")
        for i in doFilter(file, ptype[idx]):
            if ctr < STOP_SIZE:
                ctr += 1
                if ctr % 10000 == 0:
                    print("<%s>" % (ptype[idx]) + "第" + str(ctr) + "件")

                for x in i:
                    marker[x] = 1

                new_line = i + "\n"


            else:
                break
        print("*** <" + ptype[idx] + ">数据过滤处理结束 ***")

    for k,v in marker.items():
        fw.write(k+"\n")
    fw.close()


# 分词
def wakati(text, mod):
    if mod == 'jieba':
        # 用jieba分词
        return jieba.cut(text)
    elif mod == 'mecab':
        # 用mecab分词
        tagger = MeCab.Tagger("-Owakati")
        return tagger.parse(text)

def readFilter(ssfile):
    ss = []
    fr = open(ssfile, "r")
    for line in fr.readlines():
        line = line.strip()
        if line != '':
            ss.append(line)
    fr.close()
    return ss

# 读取CSV文件
def readDataFile(fileName,filter,ptype):
    
    if ptype == 'train':
        # 训练数据4个字段
        COND = 4
    elif ptype == 'test':
        # 测试数据3个字段
        COND = 3
    ferror = open(fileName + "_error.txt", 'w')

    with open(fileName,"r") as csvfile:
        reader = csv.reader(csvfile,delimiter = '\t')
        # 读取内容
        # 用csv.reader 读取 line 已经是list了
        for line in reader:
            if len(line) == COND:
                # 标题分词                               
                seg_title = wakati(line[1].strip(),'jieba')
                outTitle = []
                for w in seg_title:
                    if w not in filter:
                        outTitle.append(w)
                seg_title_return = " ".join(outTitle)
                del outTitle

                # 正文分词
                seg_content = wakati(line[2].strip(),'jieba')
                outContent = []
                for w in seg_content:
                    if w not in filter:
                        outContent.append(w)
                seg_content_return = " ".join(outContent)
                del outContent
#                 print ",".join(seg_content)
                if ptype == 'train':
                    # label = line[3]
                    yield seg_title_return,seg_content_return,line[3].strip()
                elif ptype == 'test':
                    # doc_id = line[0]
                    yield seg_title_return,seg_content_return,line[0].strip()
            else:
                ferror.write(" ".join(line) + "\n")

    ferror.close()


def run(input_file,outfile_all,trainfilter,ptype):
    # 定义训练文件
    print("*** <" + ptype + ">数据分词处理开始 ***")
    # 每次获取最大件数
    STOP_SIZE = 600001
    # 初始化ctr值
    ctr = 0
    fw = open(outfile_all, 'a')

    filter = readFilter(trainfilter)

    for i in readDataFile(input_file,filter,ptype):
        if ctr< STOP_SIZE:
            ctr +=1
            if ctr % 10000 == 0:
                print("<%s>" % (ptype) + "第" + str(ctr)+ "件")
            #去除标点和特殊符号后的标题和内容取得
            title = i[0]
            content = i[1]
            # if ptype == 'train':label = i[2]
            # if ptype == 'test':doc_id = i[2]
            if ptype == 'train':
                new_all = '__label__' + i[2] + " "+ title + " " + content+ "\n"
            elif ptype == 'test':
                new_all = i[2] + " "+ title + " " + content+ "\n"
            
            #写入新的文件
            fw.write(new_all)
            
        else:
            break

    fw.close()
    print("*** <" + ptype + ">数据分词处理结束 ***")


# Main函数
def main():
    # 训练数据
    dataPath = "/media/kinux2347/software/DataScience/bdci360_semi/"
    input_train = dataPath + "train.tsv"

    train_outfile_all = dataPath + "train/train_all.tsv"
    
    # 测试数据
    input_test = dataPath + "evaluation_public.tsv"
    
    test_outfile_all = dataPath + "test/test_all.tsv"

    trainfilter = dataPath + "filter.txt"

    # 过滤标点符号,写入文件
    # makeFilterFile([input_train,input_test], trainfilter, ['train','test'])

    # 处理训练文件
    # run(input_train,train_outfile_all,trainfilter,'train')
    
    # 处理测试文件
    run(input_test,test_outfile_all,trainfilter,'test')


if __name__ == '__main__':
    main()

