"""
生成训练数据，测试数据和验证数据集
"""

import jieba,os,csv
import jieba.posseg as psg

#源数据所在的文件夹
dir_path = os.getcwd() + os.sep + 'data' + os.sep

#创建三个文件，分别用来写入训练数据，测试数据，验证数据
train = open('dataset/train_data.txt','w',encoding='utf-8')
dev = open('dataset/dev_data.txt','w',encoding='utf-8')
test = open('dataset/test_data.txt','w',encoding='utf-8')

#创建实体类型集合
type = set(['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW','CL'])

#常见的标点符号集合
fuhao = set(['。','?','？','!','！'])

#读取字典文件，常见的实体标注
dict = csv.reader(open('dict/word_tag.csv','r',encoding='utf-8'))

#将实体词汇动态添加到jieba分词词库
for row in dict:
    if len(row) == 2:
        jieba.add_word(row[0].strip(),tag=row[1].strip())
        jieba.suggest_freq(row[0].strip())

file_name = 0 #文件数量

#遍历每个文件
for file in os.listdir(dir_path):
    if "txtoriginal.txt" in file:
        fp = open(dir_path + file,'r',encoding='utf-8')
        for line in fp:  #遍历每个文件内容
            file_name += 1
            words = psg.cut(line)
            for key,value in words:
                if key.strip() and value.strip(): #保证key和value都不为空字符串
                    #设置将数据分成三个数据集的标志,相当于将数据集分成了三份
                    if file_name % 15 < 2:
                        index = '1'
                    elif (file_name % 15) > 1 and (file_name % 15) < 4:
                        index = '2'
                    else:
                        index = '3'

                    #对每个词进行标记
                    #1.词性不在标记的词性范围内，直接打O
                    if value.strip() not in type:
                        value = 'O'
                        #遍历词语里面的每个字符
                        for achar in key.strip():
                            #如果需要标记的字符是分句常见的标点符号，则转换为标点符号 O 两个换行
                            if achar and achar.strip() in fuhao:
                                string = achar + ' ' + value.strip() + '\n' + '\n'
                                dev.write(string) if index == '1' else test.write(string) if index == '2' else train.write(string)
                            #如果不是分句位置的标点符号，则转换为标点符号 O 一个换行
                            if achar and achar.strip() not in fuhao:
                                string = achar + ' ' + value.strip() + '\n'
                                dev.write(string) if index == '1' else test.write(string) if index == '2' else train.write(string)
                    #词性在标记的词性范围内，需要在前面加上BI标记 开头的字符以B开头，后面的都为I开头
                    elif value.strip() in type:
                        begin = 0
                        #遍历词语里面的每个字符
                        for char in key.strip():
                            if begin == 0:
                                begin += 1
                                string1 = char + ' ' + 'B-' + value.strip() + '\n'
                                if index == '1':
                                    dev.write(string1)
                                elif index == '2':
                                    test.write(string1)
                                else:
                                    train.write(string1)
                            else:
                                string2 = char +' ' + 'I-' + value.strip() + '\n'
                                if index == '1':
                                    dev.write(string2)
                                elif index == '2':
                                    test.write(string2)
                                else:
                                    train.write(string2)
                    else:
                        continue
dev.close()
test.close()
train.close()










