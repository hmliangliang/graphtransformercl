#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 19:41
# @Author  : Liangliang
# @File    : graphtransformercl.py
# @Software: PyCharm

import os
os.system("pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html")
os.environ['DGLBACKEND'] = "tensorflow"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

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
import dgl
import samplegraph
from multiprocessing.pool import ThreadPool

import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

# 设置随机种子点
#random.seed(921208)
np.random.seed(921208)
tf.random.set_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"
# 设置GPU随机种子点
os.environ['TF_DETERMINISTIC_OPS'] = '1'

e = 1e-6

#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


'''
GraphTransformer结构参考:
1.基于dgl实现方法：https://www.thepaper.cn/newsDetail_forward_22203105
2.多头自注意力机制: https://zhuanlan.zhihu.com/p/410776234
3.Dwivedi V P, Bresson X. A generalization of transformer networks to graphs[C]. AAAI 2021. Fig.1
'''

#定义GraphTransformer基础层模型
class GraphTransformerLayer(keras.Model):
    def __init__(self, input_dim=256, n_heads=8, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.n_heads = n_heads
        assert input_dim % n_heads == 0
        self.head_dim = input_dim // n_heads
        self.Q_mat = keras.layers.Dense(input_dim, use_bias=False)
        self.V_mat = keras.layers.Dense(input_dim, use_bias=False)
        self.K_mat = keras.layers.Dense(input_dim, use_bias=False)
        self.cov = keras.layers.Dense(input_dim)
        self.layers_norm1 = keras.layers.BatchNormalization()
        self.layers_norm2 = keras.layers.BatchNormalization()
        self.drop_out = keras.layers.Dropout(dropout)

    def call(self, g, inputs, training=None, mask=None):
        #把邻接矩阵转为稀疏矩阵,防止把显存挤爆
        n, d = inputs.shape
        d = max(1.0 / math.sqrt(d), e)
        indices = tf.cast(tf.stack(g.edges(), axis=1), tf.int64)
        A = tf.sparse.SparseTensor(indices=indices, values=[1.0] * indices.shape[0], dense_shape=[n, n])
        Q = tf.split(self.Q_mat(inputs), self.n_heads, axis=1)
        V = tf.split(self.V_mat(inputs), self.n_heads, axis=1)
        K = tf.split(self.K_mat(inputs), self.n_heads, axis=1)
        #计算多头自注意力score
        h = tf.concat([tf.matmul(tf.nn.softmax(tf.sparse.sparse_dense_matmul(
            tf.matmul(Q[i], tf.transpose(K[i], perm=[1, 0])), A) * d, axis=1), V[i]) for i in range(self.n_heads)],
                  axis=1)
        h = inputs + h
        if training:
            h = self.layers_norm1(h)
        h1 = self.cov(h)
        if training:
            h1 = self.drop_out(h1)
        h1 = tf.nn.leaky_relu(h1)
        h = h + h1
        if training:
            h = self.layers_norm2(h)
        return h


#定义GraphTransformer模型
class GraphTransformer(keras.Model):
    def __init__(self, input_dim=256, output_dim=128, L=3, n_heads=8, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.n_heads = n_heads
        self.L = L
        self.layers_gf = [GraphTransformerLayer(input_dim, n_heads, dropout) for _ in range(L)]
        self.drop_out = keras.layers.Dropout(dropout)
        self.layer_normalization = keras.layers.BatchNormalization()
        self.cov = keras.layers.Dense(output_dim)

    def call(self, g, inputs, training=None, mask=None):
        for i in range(self.L):
            inputs = self.layers_gf[i](g, inputs, training=training)
        inputs = self.cov(inputs)
        if training:
            inputs = self.drop_out(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        if training:
            inputs = self.layer_normalization(inputs)
        return inputs


'''
关于reparameterization trick参考于
Contrastive Graph Structure Learning via Information Bottleneck for Recommendation NeurIPS2022 中的(11)式
'''
#基于对比学习的GraphTransformer模型
class GraphTransformerCL(keras.Model):
    def __init__(self, input_dim=256, feat_dim=96, output_dim=64, output_dim1=128, L=3, n_heads=8, dropout=0.1):
        super(GraphTransformerCL, self).__init__()
        self.cov = keras.layers.Dense(input_dim)
        self.GraphTransformer = GraphTransformer(input_dim, output_dim1, L, n_heads, dropout)
        self.porb_t = keras.layers.Dense(output_dim1)
        self.cov1_t = keras.layers.Dense(feat_dim)
        self.drop_out_t = keras.layers.Dropout(dropout)
        self.cov2_t = keras.layers.Dense(output_dim)
        self.cov3_t = keras.layers.Dense(output_dim)
        self.cov1_s = keras.layers.Dense(feat_dim)
        self.drop_out_s = keras.layers.Dropout(dropout)
        self.cov2_s = keras.layers.Dense(output_dim)
        self.porb_s = keras.layers.Dense(output_dim1)

    def call(self, g, inputs, training=None, mask=None):
        inputs = self.cov(inputs)
        inputs = self.GraphTransformer(g, inputs, training=training)
        if training:
            prob_t = self.porb_t(inputs)
            prob_t = tf.nn.sigmoid(prob_t)
            u_t = tf.random.uniform((prob_t.shape[0], prob_t.shape[1]))
            prob_s = self.porb_s(inputs)
            prob_s = tf.nn.sigmoid(prob_s)
            u_s = tf.random.uniform((prob_s.shape[0], prob_s.shape[1]))
            h_t = tf.nn.sigmoid((tf.math.log(u_t + e) - tf.math.log(1 - u_t + e) + prob_t) / args.tau) * inputs
            h_s = tf.nn.sigmoid((tf.math.log(u_s + e) - tf.math.log(1 - u_s + e) + prob_s) / args.tau) * inputs
        else:
            h_t = inputs
        #teacher网络的输出
        h_t = self.cov1_t(h_t)
        h_t = self.drop_out_t(h_t)
        h_t = tf.nn.leaky_relu(h_t)
        h_t = self.cov2_t(h_t)
        h_t = tf.nn.leaky_relu(h_t)
        h_t = self.cov3_t(h_t)
        h_t = tf.nn.leaky_relu(h_t)
        #student网络输出
        if training:
            h_s = self.cov1_t(h_s)
            h_s = self.drop_out_s(h_s)
            h_s = tf.nn.leaky_relu(h_s)
            h_s = self.cov2_t(h_s)
            h_s = tf.nn.leaky_relu(h_s)
            h_t = tf.nn.l2_normalize(h_t, axis=1)
            h_s = tf.nn.l2_normalize(h_s, axis=1)
            return [h_t, h_s]
        else:
            h_t = tf.nn.l2_normalize(h_t, axis=1)
            return h_t


def loss_function(h_t, h_s):
    '''
    对比学习loss采用 RINCE: Robust InfoNCE Loss
    原始文献: Chuang C Y, Hjelm R D, Wang X, et al. Robust contrastive learning against noisy views[C]
    //Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 16670-16681.
    '''
    q = args.q
    lamda = args.lamda
    n = h_t.shape[0]
    pos = tf.reduce_sum(tf.reduce_sum(h_s * h_t, axis=1)) / n
    h_t_neg = tf.repeat(h_t, args.neg_num, axis=0)
    #随机负采样
    samples = tf.concat([tf.gather(h_s, np.random.randint(0, n, (1, args.neg_num))[0], axis=0) for _ in range(n)], axis=0)
    neg = tf.reduce_sum(tf.reduce_sum(h_t_neg * samples, axis=1)) / (n * args.neg_num)
    loss = - tf.math.exp(q * pos) / q + ((lamda * (pos + neg))**q) / q
    return loss


#读取输入的图数据
def read_graph():
    '''读取图数据部分
    二维数组，每一个元素为边id，一行构成一条边
    '''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        # 读取边结构数据
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)
    # 读取属性特征信息
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        # 读取属性特征数据
        data_attr = pd.concat([data_attr, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)
    # 读取节点的属性特征数据
    data_attr = tf.convert_to_tensor(data_attr.values, dtype=tf.float32)
    # 定义图结构
    g = dgl.graph((data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()), num_nodes=data_attr.shape[0],
                  idtype=tf.int32)
    # 转化为无向图
    g = dgl.to_bidirected(g)
    g.ndata["feat"] = data_attr
    return g


#执行训练过程
def train():
    g = read_graph()
    if args.env == "train":
        model = GraphTransformerCL(args.input_dim, args.feat_dim, args.output_dim, args.output_dim1,
                                   args.L, args.n_heads, args.dropout)
    else:
        # 装载训练好的模型
        model = GraphTransformerCL(args.input_dim, args.feat_dim, args.output_dim, args.output_dim1,
                                   args.L, args.n_heads, args.dropout)
        cmd = "s3cmd get -r  " + args.model_output + "graphtransformercl"
        os.system(cmd)
        checkpoint_path = "./graphtransformercl/graphtransformercl.pb"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print("GraphTransformerCL is loaded!")
    beforeLoss = 2 ** 23
    stopNum = 0
    n = g.number_of_nodes()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    for i in range(args.sample_num):
        if i % args.batch_num == 0:
            print("开始第{}个子图的采样. {}".format(i, datetime.datetime.now()))
        g_sub = samplegraph.get_subgraph(random.randint(0, n), g, args)
        if i % args.batch_num == 0:
            print("第{}个采样的子图 节点数目:{} edges数目:{} {}".format(i, g_sub.number_of_nodes(), g_sub.number_of_edges(), datetime.datetime.now()))
        #获取拉普拉斯位置编码信息
        feat = dgl.lap_pe(g_sub, args.k_value, padding=True)
        feat = tf.math.real(feat)
        feat = tf.concat([g_sub.ndata["feat"], feat], axis=1)
        loss = 0
        with tf.GradientTape(persistent=True) as tape:
            h_t, h_s = model(g_sub, feat, training=True)
            loss = loss_function(h_t, h_s)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        #间隔打印训练过程中的loss
        if i % args.batch_num == 0:
            print("一共需采样{}个子图第{}个子图训练的loss:{} {}".format(args.sample_num, i, loss, datetime.datetime.now()))
        #判断训练是否优化
        if loss < beforeLoss:
            beforeLoss = loss
            stopNum = 0
            # 保存训练模型
            # model.summary()
            model.save_weights("./graphtransformercl/graphtransformercl.pb", save_format="tf")
            cmd = "s3cmd put -r ./graphtransformercl " + args.model_output
            os.system(cmd)
            print("在训练第{}个子图后,model模型已保存! {}".format(i, datetime.datetime.now()))
        else:
            stopNum = stopNum + 1
            if stopNum > args.stop_num:
                print("Early stop!")
                break


#判断是否是二维列表
def is_list(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return True
    else:
        return False

#把推理结果写入文件系统中
def write(data, count):
    # 注意在此业务中data是一个二维list
    # 数据的数量
    print("开始写入第{}个文件数据. {}".format(count, datetime.datetime.now()))
    n = len(data)
    flag = is_list(data[0])
    if n > 0:
        start = time.time()
        with open(os.path.join(args.data_output, 'pred_{}.csv'.format(count)), mode="a") as resultfile:
            if flag:
                for i in range(n):
                    line = ",".join(map(str, data[i])) + "\n"
                    resultfile.write(line)
            else:
                line = ",".join(map(str, data)) + "\n"
                resultfile.write(line)
        cost = time.time() - start
        print("第{}个大数据文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(count, n, cost, datetime.datetime.now()))


def node_inference(g, i, s, model, result, count):
    if i % 2000 == 0:
        #print("开始执行第{}个文件中第{}个节点的推理过程. {}".format(count, i, datetime.datetime.now()))
        start = time.time()
    g_sub = samplegraph.get_subgraph(i, g, args)
    if i % 2000 == 0:
        end = time.time()
        print("单次采样子图的耗时为:", end - start)
    feat = dgl.lap_pe(g_sub, args.k_value, padding=True)
    feat = tf.math.real(feat)
    feat = tf.concat([g_sub.ndata["feat"], feat], axis=1)
    h = model(g_sub, feat, training=False)
    j = tf.where(g_sub.ndata[dgl.NID] == i)[0, 0]
    result[s, 0] = str(i)
    result[s, 1:] = h[j, :].numpy().astype("str")
    if i % 2000 == 0:
        print("第{}个文件中第{}个节点的推理过程执行完成!. {}".format(count, i, datetime.datetime.now()))

#单次推理过程
def inference_step(g, nodes, model, count):
    n = len(nodes)
    result = np.zeros((n, args.output_dim + 1)).astype("str")
    s = -1
    #pool = ThreadPool(args.workers)
    for i in nodes:
        s = s +1
        node_inference(g, i, s, model, result, count)
        #pool.apply_async(node_inference, args=(g, i, s, model, result, count,))
    #pool.close()  # 关闭进程池
    #pool.join()
    print("准备开始写入第{}个文件输出数据. {}".format(count, datetime.datetime.now()))
    write(result.tolist(), count)


#执行推理过程
def inference():
    g = read_graph()
    n = g.number_of_nodes()
    # 装载训练好的模型
    model = GraphTransformerCL(args.input_dim, args.feat_dim, args.output_dim, args.output_dim1,
                               args.L, args.n_heads, args.dropout)
    cmd = "s3cmd get -r  " + args.model_output + "graphtransformercl"
    os.system(cmd)
    checkpoint_path = "./graphtransformercl/graphtransformercl.pb"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    '''
    对比学习推理只会用到teacher或student网络中的一个,加上expect_partial()主要忽略未使用的权值而产生的大量警告
    '''
    model.load_weights(latest).expect_partial()
    print("GraphTransformerCL is loaded!")
    #每次读取一个文件的数据
    n_files = math.ceil(n / args.file_nodes_max_num)
    for i in range(n_files):
        nodes = [j for j in range(i*args.file_nodes_max_num, min((i+1)*args.file_nodes_max_num, n))]
        print("一共{}个任务开始分发第{}个推理子任务 {}".format(n_files, i, datetime.datetime.now()))
        inference_step(g, nodes, model, i)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or inference)", type=str, default='train_incremental')
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--dropout", help="dropout比率", type=float, default=0.15)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=5000)
    parser.add_argument("--workers", help="设置多线程的线程数目", type=int, default=10)
    parser.add_argument("--neg_num", help="采样子图的节点数目", type=int, default=5)
    parser.add_argument("--n_heads", help="多头自注意力机制的头数", type=int, default=8)
    parser.add_argument("--k_value", help="选取拉普拉斯位置编码方法的非零最小特征值数目", type=int, default=16)
    parser.add_argument("--L", help="transformer结构的层数", type=int, default=3)
    parser.add_argument("--sample_num", help="采样子图的子图数目", type=int, default=20000)
    parser.add_argument("--batch_num", help="打印loss函数的周期", type=int, default=100)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=256)
    parser.add_argument("--feat_dim", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim1", help="隐含层特征的维度", type=int, default=128)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--subgraph_nodes_max_num", help="采样子图最大的节点数目", type=int, default=30)
    parser.add_argument("--subgraph_edges_max_num", help="采样子图最大的边数目", type=int, default=1500)
    parser.add_argument("--subgraph_nodes_min_num", help="采样子图最小的节点数目", type=int, default=18)
    parser.add_argument("--subgraph_edges_min_num", help="采样子图最小的边数目", type=int, default=17)
    parser.add_argument("--k_hop", help="采样子图的跳连数目", type=int, default=1)
    parser.add_argument("--tau", help="计算reparameterization trick的temperature", type=float, default=1.1)
    parser.add_argument("--lamda", help="loss中的lamda参数", type=float, default=0.025)
    parser.add_argument("--q", help="loss中的q参数", type=float, default=0.5)
    parser.add_argument("--file_nodes_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=10000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://general__lingqu/xxx/models/graphtransformercl/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        train()
    elif args.env == "inference":
        inference()
    else:
        raise TypeError("args.env必需是train或train_incremental或inference！")
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))
