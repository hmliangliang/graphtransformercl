# -*-coding: utf-8 -*-
# @Time    : 2023/6/30 17:31
# @Author  : Liangliang
# @File    : dnnrank.py
# @Software: PyCharm

import os
import datetime
import argparse
import time
import math
import random
import s3fs
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# 设置随机种子点
random.seed(921208)
np.random.seed(921208)
tf.random.set_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"
# 设置GPU随机种子点
os.environ['TF_DETERMINISTIC_OPS'] = '1'

e = 1e-6


# 读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


# 关于Hidden Gate机制参考于：https://zhuanlan.zhihu.com/p/523185088
class dnn_rank_gate(keras.Model):
    def __init__(self, input_dim, feat_dim, output_dim):
        super(dnn_rank_gate, self).__init__()
        self.cov1 = tf.keras.layers.Dense(input_dim)
        self.hidden_gate1 = tf.keras.layers.Dense(input_dim, use_bias=False)
        self.cov2 = tf.keras.layers.Dense(feat_dim)
        self.hidden_gate2 = tf.keras.layers.Dense(feat_dim, use_bias=False)
        self.cov3 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        inputs = self.cov1(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        h = self.hidden_gate1(inputs)
        h = tf.nn.sigmoid(h)
        inputs = h * inputs
        inputs = self.cov2(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        h = self.hidden_gate2(inputs)
        h = tf.nn.sigmoid(h)
        inputs = h * inputs
        inputs = self.cov3(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = tf.nn.softmax(inputs)
        inputs = tf.reshape(inputs[:, 1], [-1, 1])
        return inputs


# 不包含gate机制以便于实验对比
class dnn_rank(keras.Model):
    def __init__(self, input_dim, feat_dim, output_dim):
        super(dnn_rank, self).__init__()
        self.cov1 = tf.keras.layers.Dense(input_dim)
        self.cov2 = tf.keras.layers.Dense(feat_dim)
        self.cov3 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        inputs = self.cov1(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.cov2(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.cov3(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = tf.nn.softmax(inputs)
        inputs = tf.reshape(inputs[:, 1], [-1, 1])
        return inputs


#定义loss函数
def loss_function(label, predict):
    '''
    label: 真实的类标签
    predict: 预测得分
    :return: loss
    '''
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, 2)
    temp = 1 - predict
    predict = tf.concat([temp, predict], axis=1)
    loss = 1 / args.batch_size * tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict)))
    return loss

#训练过程
def train():
    # 读取数据文件
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    n_files = len(input_files)
    # 定义神经网络结构
    if args.env == "train":  # 第一次训练,创建模型
        model = dnn_rank(args.input_dim, args.feat_dim, args.output_dim)
    elif args.env == "train_gate":  # 第一次训练,创建模型
        model = dnn_rank_gate(args.input_dim, args.feat_dim, args.output_dim)
    elif args.env == "train_incremental":
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "dnnrank"
        os.system(cmd)
        model = tf.keras.models.load_model("./dnnrank", custom_objects={'tf': tf}, compile=False)
        print("dnn_rank Model weights load!")
    else:  # 利用上一次训练好的模型，进行增量式训练
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "dnnrankgate"
        os.system(cmd)
        model = tf.keras.models.load_model("./dnnrankgate", custom_objects={'tf': tf}, compile=False)
        print("dnn_rank_gate Model weights load!")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    print("开始读取数据! {}".format(datetime.datetime.now()))
    before_loss = 2 ** 32
    stop_num = 0
    loss = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    for epoch in range(args.epoch):
        count = 0
        for file in input_files:
            count = count + 1
            print("epoch:{} 一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(
                epoch, n_files, count, "s3://" + file))
            # 读取训练数据数据,不含节点的id
            data = pd.read_csv("s3://" + file, sep=',', header=None)
            data = tf.convert_to_tensor(data.values, dtype=tf.float32)
            label = tf.reshape(data[:, -1], (-1, 1))
            data = data[:, :-1]
            data = tf.data.Dataset.from_tensor_slices((data, label)).shuffle(100).batch(args.batch_size,
                                                                                        drop_remainder=True)
            count_batch = 0
            for batch_data, batch_label in data:
                count_batch = count_batch + 1
                with tf.GradientTape(persistent=True) as tape:
                    predict = model(batch_data)
                    loss = loss_function(batch_label, predict)
                    #打印loss，评测训练数据
                    if random.random() <= 0.01:
                        auc = roc_auc_score(batch_label, predict)
                        predict_label = tf.where(predict >= 0.5, 1.0, 0.0)
                        pos = tf.reduce_sum(batch_label) / len(batch_label)
                        pos_pred = tf.reduce_sum(predict_label) / len(predict_label)
                        print("epoch:{} 第{}个文件 第{}个batch的预测结果 loss:{} 之前最好的loss:{} 真实正样本比例:{} "
                              "预测正样本比例:{} auc:{} {}".format(epoch, count, count_batch, loss, before_loss,
                                                                   pos, pos_pred, auc, datetime.datetime.now()))
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if loss < before_loss:
            before_loss = loss
            stop_num = 0
            # 保存模型model
            # net.summary()
            if args.env == "train" or args.env == "train_incremental":  # dnn_rank模型
                model.save("./dnnrank", save_format="tf")
                print("dnn_rank模型已保存!")
                cmd = "s3cmd put -r ./dnnrank " + args.model_output
                os.system(cmd)
            elif args.env == "train_gate" or args.env == "train_incremental_gate":  # dnn_rank_gate模型
                model.save("./dnnrankgate", save_format="tf")
                print("dnn_rank_gate已保存!")
                cmd = "s3cmd put -r ./dnnrankgate " + args.model_output
                os.system(cmd)
            else:
                raise TypeError("args.env设置不符合要求!")
        else:
            stop_num = stop_num + 1
            if stop_num > args.stop_num:
                print("Early stop! {}".format(datetime.datetime.now()))
                break


def inference():
    if args.env == "inference":  # dnn_rank模型
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "dnnrank"
        os.system(cmd)
        model = tf.keras.models.load_model("./dnnrank", custom_objects={'tf': tf}, compile=False)
        print("dnn_rank Model weights load!")
    elif args.env == "inference_gate":  # dnn_rank_gate模型
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "dnnrankgate"
        os.system(cmd)
        model = tf.keras.models.load_model("./dnnrankgate", custom_objects={'tf': tf}, compile=False)
        print("dnn_rank_gate Model weights load!")
    else:
        raise TypeError("args.env设置不符合要求!")
    #读取文件数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    n_data_files = len(input_files)
    for file in input_files:
        count = count + 1
        print("一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(n_data_files, count, "s3://" + file))
        # 读取训练数据数据,第0列为节点的id
        data = pd.read_csv("s3://" + file, sep=',', header=None).astype(object)
        n = data.shape[0]
        n_files = math.ceil(n / args.file_nodes_max_num)
        for i in range(n_files):
            data_batch = data.iloc[i * args.file_nodes_max_num:min((i + 1)*args.file_nodes_max_num, n), 2:]
            # ID为一个pair的ID编号
            ID = data.iloc[i * args.file_nodes_max_num:min((i + 1)*args.file_nodes_max_num, n), 0:2]
            result = np.zeros((ID.shape[0], 3)).astype('str')
            data_batch = tf.convert_to_tensor(data_batch.values, dtype=tf.float32)
            data_batch = model(data_batch)
            result[:, 0:2] = ID.values
            result[:, 2] = data_batch.numpy()[:, 0].astype('str')
            write(result.tolist(), count, i)


# 判断是否是二维列表
def is_list(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return True
    else:
        return False


def write(data, count, i):
    n = len(data)
    flag = is_list(data[0])
    line = ""
    if n > 0:
        start = time.time()
        with open(os.path.join(args.data_output, 'pred_{}_{}.csv'.format(count, i)), mode="a") as resultfile:
            if flag:
                for j in range(n):
                    line = line + ",".join(map(str, data[j])) + "\n"
                resultfile.write(line)
            else:
                line = ",".join(map(str, data)) + "\n"
                resultfile.write(line)
        cost = time.time() - start
        print("第{}个大数据文件的第{}个子文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(
            count, i, n, cost, datetime.datetime.now()))


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or inference)", type=str, default='train_incremental_gate')
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--epoch", help="学习率", type=int, default=100)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=50)
    parser.add_argument("--batch_size", help="batch_size大小", type=int, default=2048)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=400)
    parser.add_argument("--feat_dim", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=2)
    parser.add_argument("--file_nodes_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=50000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://general__lingqu/xxx/lgames/dnn/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env.find("train") != -1:
        train()
    elif args.env.find("inference") != -1:
        inference()
    else:
        raise TypeError("args.env必需是train或train_incremental或inference！")
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))
