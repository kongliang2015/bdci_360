# -*- encoding: utf-8 -*-
import sys
import csv

reload(sys)
sys.setdefaultencoding('utf-8')
csv.field_size_limit(sys.maxsize)


# 读取CSV文件
def readDataFile(fileName,ptype):

    with open(fileName,"r") as csvfile:
        reader = csv.reader(csvfile,delimiter = ' ')
        # 读取内容
        # 用csv.reader 读取 line 已经是list了
        for line in reader:
            yield line[0].strip()

def run(input_file,outfile_all,ptype):

    # 每次获取最大件数
    STOP_SIZE = 600001
    # 初始化ctr值
    ctr = 0
    fw = open(outfile_all, 'a')

    for i in readDataFile(input_file,ptype):
        if ctr< STOP_SIZE:
            ctr +=1
            #去除标点和特殊符号后的标题和内容取得
            new_all = i + "\n"
            #写入新的文件
            fw.write(new_all.encode("utf-8"))

        else:
            break

    fw.close()

# Main函数
def main():
    # 训练数据
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360-Semi/"

    testText = dataPath + "test/test_all.tsv"

    test_outfile_all = dataPath + "test/test_all_docid.txt"
    
    # 处理测试文件
    run(testText,test_outfile_all,'test')
    

if __name__ == '__main__':
    main()

