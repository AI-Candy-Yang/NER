"""
用来进行数据处理的文件
"""
import codecs
import re
import jieba
import random
# def get_sentence_int_to_vocab(train_path,test_path,dev_path):
#     sentence_vocab = []
#     f1 = open(train_path,encoding="utf-8")
#     f2 = open(test_path,encoding="utf-8")
#     f3 = open(dev_path,encoding="utf-8")

#将数据转换成列表
def load_sentences(path,zero):
    sentences = []
    sentence = []
    for line in codecs.open(path,'r','utf-8'):
        if zero: #如果为数字，则用数字替换
            line = re.sub('\d','0',line.rstrip())
        else:
            line = line.rstrip()

        if not line: #为空行的时候
            if len(sentence) > 0:
                sentences.append(sentence)
        else:
            word = line.split( )
            assert len(word) == 2
            sentence.append(word)
    #添加最后一个句子
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

#文字数字化，统计每个字频
def creat_dict(chars):
    result_dict = {}
    for items in chars:
        for item in items:
            if item not in result_dict.keys():
                result_dict[item] = 1
            else:
                result_dict[item] += 1
    return result_dict

#根据字频创建索引字典
def creat_mapping(dictionary):
    #字典根据值进行排序
    sorted_items = sorted(dictionary.items(),key=lambda x:x[1],reverse=True)
    id_to_item = {i : v[0] for i,v in enumerate(sorted_items)}
    item_to_id = {v : k for k,v in id_to_item.items()}
    return item_to_id,id_to_item


#构建字语料库字典，字--索引字典   索引-字字典
def char_mapping(sentences,lower):
    #将每个字符转为小写
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    #根据字频创建语料库字典 键是字，值是字频
    dictionary = creat_dict(chars)
    dictionary['<PAD>'] = 10000001
    dictionary['<UNK>'] = 10000000
    #根据字频高低进行排序，得到每个字的索引
    char_to_id,id_to_char = creat_dict(dictionary)

    return dictionary,char_to_id,id_to_char

#构建标签语料库字典，标签--索引字典  索引--标签字典,并写入文件
def tag_mapping(sentences):
    f = open('tag_to_id.txt','w',encoding='utf-8')
    f1 = open('id_to_tag','w',encoding='utf-8')

    tags = [[char[-1] for char in s] for s in sentences]
    #统计标签频率
    tag_dict = creat_dict(tags)

    #根据频率创建标签--索引字典
    tag_to_id,id_to_tag = creat_mapping(tag_dict)

    #将两个字典数据写入文件
    for k,v in tag_to_id.items():
        f.write(k + ":" + str(v) + '\n')

    for k,v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")

    return tag_dict,tag_to_id,id_to_tag

#将训练文本进行分词，并且用数字进行标注 单个字用0标注，超过两个字的，第一个用1，中间的用2，最后一个用3进行标注
def get_seg_features(string):
    seg_feature = []
    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)

    return seg_feature

#将训练数据转换成数字向量
def prepare_dataset(sentences,char_to_id,tag_to_id,lower=False,train=True):
    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        #将训练数据根据词典文件转化为数字索引
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>'] for w in string]

        #将训练数据转化成数字标签
        segs = get_seg_features(string)
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        # data.append([string,chars,segs,tags])
        data.append([chars,tags])

    return data

class Batch:
    def __init__(self):
        self.inputs = []
        self.inputs_length = []
        self.targets = []
        self.targets_length = []

#创建batch数据
def createBatch(samples):
    batch = Batch()
    batch.inputs_length = [len(sample[0]) for sample in samples]
    batch.targets_length = [len(sample[1]) for sample in samples]
    max_source_length = max(batch.inputs_length)
    max_target_length = max(batch.targets_length)

    for j,sample in enumerate(samples):
        source = sample[0]
        batch.inputs.append(source + [0]*(max_source_length-len(source)))
        target = sample[1]
        batch.targets.append(target + [0]*(max_target_length-len(target)))

    return batch

#获取batch数据
def getBatches(data,batch_size):
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def getNextSamples():
        for i in range(0,data_len,batch_size):
            yield data[i:min(i+batch_size,data_len)]

    for sample in getNextSamples():
        batch = createBatch(sample)
        batches.append(batch)
    return batches



    
    






