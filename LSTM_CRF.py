"""
LSTM_CRF模型文件
"""
import tensorflow as tf
import tensorflow.contrib as ct
import numpy as np
import sklearn.metrics as sk

class LSTM_CRFModel():
    #模型参数初始化函数
    def __init__(self,rnn_size,embedding_size,learning_rate,sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vacab,keep_prob):
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.vocab_size = len(sentence_vocab_to_int)
        self.tag_size = len(tags_vocab_to_int)
        self.keep_prob = keep_prob
        self.idx_to_tar = tags_int_to_vacab
        self.buildmodel()

    #构建模型的网络结构 bilstm--全连接--CRF--计算准确率--优化器
    def buildmodel(self):
        self.inputs = tf.placeholder(tf.int32,[None,None],name = 'inputs')
        self.targets = tf.placeholder(tf.int32,[None,None],name = 'targets')
        self.inputs_length = tf.placeholder(tf.int32,[None],name = 'inputs_length')
        self.targets_length = tf.placeholder(tf.int32,[None],name='targets_length')
        self.batch_size = tf.placeholder(tf.int32,[],name = 'batch_size')

        #bilstm
        with tf.variable_scope("bilstm"):
            cell = ct.rnn.LSTMCell(self.rnn_size)
            cell = ct.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
            #tf.get_variable 获取一个存在的变量或创建一个新的变量 xavier_initializer初始化权重矩阵，用来保持每一层的梯度大小都差不多
            embedding_matrix = tf.get_variable("embedding_matrix",dtype=tf.float32,initializer=ct.layers.xavier_initializer(),shape=[self.vocab_size,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding_matrix,self.inputs)
            #构建双向的lstm模型，得到输出和隐层状态
            outputs , state = tf.nn.bidirectional_dynamic_rnn(cell,cell,inputs,self.input_length,dtype=tf.float32)
            #将前向和后向lstm输出进行拼接
            outputs = tf.concat(outputs,2)

        #全连接层
        with tf.variable_scope("dense"):
            outputs = tf.reshape(outputs,[-1,2 * self.rnn_size])
            w_dense = tf.get_variable("w_dense",dtype=tf.float32,initializer=ct.layers.xavier_initializer(),
                                      shape = [2 * self.rnn_size,self.tag_size])
            b_dense = tf.get_variable("b_dense",dtype=tf.float32,initializer=tf.zeros_initializer,shape=[self.tag_size])
            outputs = tf.matmul(outputs,w_dense) + b_dense
            outputs = tf.reshape(outputs,[self.batch_size,-1,self.tag_size])

        #CRF层
        with tf.variable_scope("CRF"):
            #crf_log_likelihood crf里面计算标签的极大似然函数,trans是转移矩阵
            log_likelihood,trans = ct.crf.crf_log_likelihood(
                outputs,self.targets,self.input_length
            )

            #得到损失函数
            self.loss = tf.reduce_mean(-log_likelihood)


        #计算准确率
        with tf.variable_scope("acc"):
            #sequence_mask 形成 true 和 false的数组
            mask = tf.sequence_mask(self.input_length)
            #crf_decode 返回最好的标签序列，viterbi_seq是最好的序列标记，viterbi_score 每个序列解码标签的分数
            viterbi_seq,viterbi_score = ct.crf.crf_decode(outputs,trans,self.input_length)
            #boolean_mask 挑出true对应的数字
            output = tf.boolean_mask(viterbi_seq,mask)
            label = tf.boolean_mask(self.targets,mask)
            # 逐个元素判断是否相等，相等就返回True,不相等就返回False
            correct_predictions = tf.equal(tf.cast(output,tf.int32),label)
            #计算准确率
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32),name = 'accuracy')
            self.pred = tf.reshape(viterbi_seq,[-1,1])

        #构建summary
        with tf.variable_scope("summary"):
            tf.summary.scalar("trainloss",self.loss)
            tf.summary.scalar("acc",self.accuracy)
            self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=None))

        #优化
        with tf.variable_scope("optimize"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            #tf.gradients 实现损失函数对参数的求导
            gradients = tf.gradients(self.loss,trainable_params)
            #tf.clip_by_global_norm 梯度裁剪
            clip_gradients,_ = tf.clip_by_global_norm(gradients,5.0)
            #优化器进行优化
            self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))
        #实例化一个Saver对象，训练网络之后保存训练好的模型，以及在程序中读取已保存好的模型
        self.saver = tf.train.Saver(tf.global_variables())


    #模型训练
    def train(self,sess,batch):
        #喂入数据
        feed_dict = {
            self.inputs : batch.inputs,
            self.inputs_length : batch.inputs_length,
            self.targets : batch.targets,
            self.targets_length : batch.targets_length,
            self.batch_size : len(batch.inputs)
        }

        _,loss,summary,acc = sess.run([self.train_op,self.loss,self.summary_op,self.accuracy],feed_dict=feed_dict)
        return loss,acc,summary

    #模型验证
    def vali(self,sess,batch):
        feed_dict = {
            self.inputs : batch.inputs,
            self.inputs_length : batch.inputs_length,
            self.targets : batch.targets,
            self.targets_length : batch.targets_length,
            self.batch_size : len(batch.inputs)
        }

        pres, loss, summary, acc = sess.run([self.pred,self.loss,self.summary_op,self.accuracy],feed_dict=feed_dict)

        labels = []
        preds = []
        for i,label in enumerate(batch.targets):
            labels.append(label[:batch.targets_length[i]])

        pred = np.reshape(preds,[len(batch.inputs),-1,1])
        for i,p in enumerate(pred):
            preds.append(p[:batch.targets_length[i]])

        labels = [self.idx_to_tar[ii] for lab in labels for ii in lab]
        preds = [self.idx_to_tar[j] for lab_pred in preds for j in lab_pred]

        _,_,micro_f1,_ = sk.precision_recall_fscore_support(labels,preds,average="micro")
        _,_,f1,_ = sk.precision_recall_fscore_support(labels,preds,average="weighted")

        return acc,loss,summary,f1,micro_f1










