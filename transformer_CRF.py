import tensorflow as tf
import numpy as np
import sklearn.metrics as sk
import tensorflow.contrib as ct

class Transformer_CRFModel():
    def __init__(self,num_heads,num_blocks,rnn_size,embedding_size,learning_rate,sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vacab,pos_vacab_to_int,keep_prob):
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.rnn_size = rnn_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.vocab_size = len(sentence_vocab_to_int)
        self.tag_size = len(tags_vocab_to_int)
        self.keep_prob = keep_prob
        self.idx_to_tar = tags_int_to_vacab
        self.pos_vocab_size = len(pos_vacab_to_int)
        self.buildmodel()

    #归一化层
    def _layerNormalization(self,inputs,scope="layerNorm"):
        epsilon = 1e-8
        inputShape = inputs.get_shape()
        paramsShape = inputShape[-1:]

        #LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有的维度
        mean,variance = tf.nn.moments(inputs,[-1],keep_dims=True)
        #参数初始化
        beta = tf.Variable(tf.zeros(paramsShape))
        gamma = tf.Variable(tf.ones(paramsShape))

        #归一化
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        #输出
        outputs = gamma * normalized + beta
        return outputs

    #多头注意力层
    def _mutilheadAttention(self,rawKeys,queries,keys,numUnits=None,causality=False,scope="multiheadAttention"):
        numHeads = self.num_heads
        keepProb = 0.9
        if numUnits is None:  #如果没有设定隐层神经元数，直接取输入的最后一维
            numUnits = queries.get_shape().as_list()[-1]

        #对输入做线性转换 每个的维度都是[batch_size,sequence_length,embedding_size]
        Q = tf.layers.dense(queries,numUnits,activation=tf.nn.relu)
        K = tf.layers.dense(keys,numUnits,activation=tf.nn.relu)
        V = tf.layers.dense(keys,numUnits,activation=tf.nn.relu)

        #将数据按最后一维分割成num_heads个，然后按照第一维度拼接
        #Q,K,V的维度都是[batch_size * num_heads,seqence_length,embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q,numHeads,axis=-1),axis=0)
        K_ = tf.concat(tf.split(K,numHeads,axis=-1),axis=0)
        V_ = tf.concat(tf.split(V,numHeads,axis=-1),axis=0)

        #计算k和q之间的点积，维度[batch_size * numheads,queries_len,key_len] 后两者为q和k的序列长度
        similary = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))

        #对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)


    #前馈神经网络
    def _feedForward(self,inputs,filters,scope="feedForward"):

        #内层
        params = {"inputs":inputs,"fliters":filters[0],"kernel_size":1,
                  "activation":tf.nn.relu,"use_bias":True}
        outputs = tf.layers.conv1d(**params)

        #外层
        params = {"inputs":outputs,"filters":filters[1],"kernel_size":1,
                  "activation":None,"use_bias":True}

        outputs = tf.layers.conv1d(**params)

        #残差连接
        outputs += inputs

        #归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs



    #位置向量
    def _positionEmbedding(self,scope="positionEmbedding"):
        batchSize = self.batch_size
        sequenceLen = 30
        embeddingSize = self.embedding_size

        #生成位置的索引向量，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen),0),[batchSize,1])


        #根据正弦和余弦函数来获取每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i - i % 2) / embeddingSize) for i in range(embeddingSize)]
                                      for pos in range(sequenceLen)])

        #然后根据奇偶性分别用sin cos函数来包装
        positionEmbedding[:,0::2] = np.sin(positionEmbedding[:,0::2])
        positionEmbedding[:,1::2] = np.cos(positionEmbedding[:,1::2])

        #将位置向量转化为tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding,dtype=tf.float32)

        #得到三维度的矩阵[batch_size,sequenceLen,embeddingSize]
        positionEmbedd = tf.nn.embedding_lookup(positionEmbedding_,positionIndex)

        return positionEmbedd

    #模型构建
    def buildmodel(self):
        self.inputs = tf.placeholder(tf.int32,[None,None],name="inputs")
        self.targets = tf.placeholder(tf.int32,[None,None],name="targets")
        self.input_length = tf.placeholder(tf.int32,[None],name="inputs_length")
        self.targets_length = tf.placeholder(tf.int32,[None],name='targets_length')
        self.pos = tf.placeholder(tf.int32,[None,None],name="pos")
        self.batch_size = tf.placeholder(tf.int32,[],name="batch_size")


        #编码阶段
        with tf.variable_scope("encoder"):
            cell = ct.rnn.LSTMCell(self.rnn_size)
            cell = ct.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
            embedding_matrix = tf.get_variable("embedding_matrix",dtype=tf.float32,initializer=ct.layers.xavier_initializer(),shape=[self.vocab_size,self.embedding_size])
            inputs = tf.nn.embedding_lookup(embedding_matrix,self.inputs)
            pos_embedding_matrix = tf.get_variable("pos_embedding_matrix",dtype=tf.float32,initializer=ct.layers.xavier_initializer(),shape=[self.pos_vocab_size,self.embedding_size])
            pos = tf.nn.embedding_lookup(pos_embedding_matrix,self.pos)

            #Bert里面是直接相加，不是拼接
            inputs = tf.concat([inputs,pos],axis=-1)
            inputs = tf.reshape(inputs,[self.batch_size,-1,2*self.embedding_size])

            with tf.variable_scope("transformer"):
                for i in range(self.num_blocks):
                    with tf.name_scope("transformer-{}".format(i+1)):
                        multiHeadAtt = self._mutilheadAttention(rawKeys=inputs,queries=inputs,keys=inputs,numUnits=None,
                                                                causality=False,scope="mutilheadAttention")

                        self.embeddedWords = self._feedForward(multiHeadAtt,2*self.rnn_size,2*self.embedding_size)

                outputs = self.embeddedWords

            with tf.variable_scope("dense"):
                outputs = tf.reshape(outputs,[-1,2*self.rnn_size])
                w_dense = tf.get_variable("w_dense",dtype=tf.float32,initializer=ct.layers.xavier_initializer(),
                                          shape=[2*self.rnn_size,self.tag_size])
                b_dense = tf.get_variable("b_dense",dtype=tf.float32,initializer=tf.zeros_initializer(),
                                          shape=[self.tag_size])
                outputs = tf.matmul(outputs,w_dense) + b_dense
                outputs = tf.reshape(outputs,[self.batch_size,-1,self.tag_size])


            #CRF层
            with tf.variable_scope("CRF"):
                log_likelihood,trans = ct.crf.crf_log_likelihood(
                    outputs,self.targets,self.input_length
                )
                self.loss = tf.reduce_mean(-log_likelihood)

            #计算准确率
            with tf.variable_scope("acc"):
                mask = tf.sequence_mask(self.input_length)
                viterbi_seq,viterbi_score = ct.crf.crf_decode(outputs,trans,self.input_length)
                output = tf.boolean_mask(viterbi_seq,mask)
                label = tf.boolean_mask(self.targets,mask)
                correct_predictions = tf.equal(tf.cast(output,tf.int32),label)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32),name="accuracy")
                self.pred = tf.reshape(viterbi_seq,[-1,])

            with tf.variable_scope("summary"):
                tf.summary.scalar("trainloss",self.loss)
                tf.summary.scalar("acc",self.accuracy)
                self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=None))

            with tf.variable_scope("opyimize"):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss,trainable_params)
                clip_gradients,_ = tf.clip_by_global_norm(gradients,5.0)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients,trainable_params))

            self.saver = tf.train.Saver(tf.global_variables())

        def train(self,sess,batch):
            feed_dict = {
                self.inputs:batch.inputs,
                self.inputs_length:batch.inputs_length,
                self.targets:batch.targets,
                self.targets_length:batch.targets_length,
                self.pos:batch.pos,
                self.batch_size:len(batch.inputs)
            }

            _,loss,summary,acc = sess.run([self.train_op,self.loss,self.summary_op,self.accuracy],feed_dict=feed_dict)
            return loss,acc,summary

        #测试
        def vali(self,sess,batch):
            feed_dict = {
                self.inputs : batch.inputs,
                self.inputs_length:batch.inputs_length,
                self.targets:batch.targets,
                self.tarfets_length:batch.targets_length,
                self.pos:batch.pos,
                self.batch_size:len(batch.inputs)
            }

            pred,loss,summary,acc = sess.run([self.pred,self.loss,self.summary_op,self.accuracy],feed_dict=feed_dict)

            labels = []
            preds = []

            for i,label in enumerate(batch.targets):
                labels.append(label[:batch.targets_length[i]])

            pred = np.reshape(pred,[len(batch.inputs),-1,1])
            for i,p in enumerate(pred):
                preds.append(p[:batch.targets_length[i]])

            labels = []
















