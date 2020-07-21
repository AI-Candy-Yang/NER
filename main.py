import tensorflow as tf
import os
from data_utils import *
import pickle
from LSTM_CRF import LSTM_CRFModel
from tqdm import tqdm  #python 进度条库

dir_path = os.getcwd() + os.sep + 'dataset' + os.sep
#定义命令行参数  变量名   变量值   变量定义
flags = tf.flags
flags.DEFINE_integer("batch_size",30,'Batch Size')
flags.DEFINE_integer("embedding_size",100,"Embedding dimension")
flags.DEFINE_float("learning_rate",0.01,"Learning_rate")
flags.DEFINE_float("keep_prob",0.5,"keep_prob")
flags.DEFINE_integer("numEpochs",30,"Maxinum of training epochs")
flags.DEFINE_string("model_dir","saves/","Path to save model checkpoints")
flags.DEFINE_string("model_name","ner.ckpt","File naem used for model checkpoints")
flags.DEFINE_string("train_file",dir_path+"train_data.txt", 'Path for train data')
flags.DEFINE_string("test_file",dir_path + "test_data.txt","Path for test data")
flags.DEFINE_string("dev_file",dir_path+"dev_data.txt",'Path for dev data')
flags.DEFINE_string("zero",True,"Wither replace digits with zero")
flags.DEFINE_string("lower",False,"Wither lower case")
flags.DEFINE_string("map_file","map.pkl","file for maps")
FLAGS = flags.FLAGS

# 1.加载数据
train_sentences = load_sentences(FLAGS.train_file, FLAGS.zero)
dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.zero)
test_sentences = load_sentences(FLAGS.test_file, FLAGS.zero)

# 2.创建字语料库字典，字---索引字典，索引---字字典
char_dict, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

# 创建标签语料库字典，标签---索引字典，索引---标签字典
tag_dict, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# 保存字典
with open(FLAGS.map_file, "wb") as f:
    pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)

def train():
    # 将训练数据转换成数字向量
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)

    with tf.Session() as sess:
        model = LSTM_CRFModel(FLAGS.rnn_size,FLAGS.embedding_size,FLAGS.learning_rate,char_to_id,tag_to_id,id_to_tag,FLAGS.keep_prob)

        #获取模型参数
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

        #如果存在就直接加载
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("reloading model parameters....")
            #Saver类训练完后，是以checkpoints文件形式保存的，提取的时候也是从checkpoints文件中恢复变量
            model.saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print("creat new model parameters")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        current_step = 0
        #验证集f1值列表
        avefllist = []
        #微平均f1值列表
        micfllist = []

        train_summary_writer = tf.summary.FileWriter("saves/train",graph=sess.graph)

        for e in range(FLAGS.numEpochs):
            print("Epoch{}/{}-----------".format(e+1,FLAGS.numEpochs))
            train_batches = getBatches(train_data,FLAGS.batch_size)
            vali_batches = getBatches(dev_data,FLAGS.batch_size)
            for train_Batch in tqdm(train_batches,desc='Training'):
                trainloss,acc,trainsummary = model.train(sess,train_Batch)
                current_step += 1
                tqdm.write("-----Step %d --trainloss %.2f --acc %.2f " %(current_step,trainloss,acc))
                train_summary_writer.add_summary(trainsummary,current_step)

                #每训练100个batch,保存一次，验证一次
                if (current_step % 100 == 0):
                    ave_f1 = 0
                    mic_f1 = 0
                    vali_current_step = 0
                    checkpoint_path = os.path.join(FLAGS.model_dir,FLAGS.model_name)

                    model.saver.save(sess,checkpoint_path,global_step=current_step)
                    for vali_Batch in tqdm(vali_batches,desc="Vali"):
                        vali_current_step += 1
                        vali_acc,valiloss,valisummary,vali_f1,mirco_f1 = model.vali(sess,vali_Batch)

                        tqdm.write("----Step %d--valiloss %.2f --valiacc %.3f --valif1 %.3f --micf1 %.3f "%(vali_current_step,valiloss,vali_acc*100,vali_f1*100,mic_f1*100))

                        ave_f1 += vali_f1
                        mic_f1 += mirco_f1

                    #计算平均值
                    ave_f1 = ave_f1/len(vali_batches)
                    mic_f1 = mic_f1/len(vali_batches)

                    avefllist.append(ave_f1)
                    micfllist.append(mic_f1)

    print("最大的avef1为：",max(avefllist))
    print("最大avef1索引为：",avefllist.index(max(avefllist)))
    print("最大micf1为：",max(micfllist))
    print("最大micf1索引为：",micfllist.index(max(micfllist)))


#模型测试
def predict():
    with open(FLAGS.map_file,"rb") as f:
        char_to_id,id_to_char,tag_to_id,id_to_tag = pickle.load(f)





# def main():
#     if FLAGS.train:
#         train()
#     else:









