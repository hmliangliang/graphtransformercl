#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 11:16
# @Author  : Liangliang
# @File    : samplegraph.py
# @Software: PyCharm


#主要作用是对子图进行采样,抽取符合要求的子图

#import tensorflow as tf
import dgl
#import time


#采样子图函数
def get_subgraph(i, g, args):
    #采样第i个节点的子图
    nodes = g.successors(i).numpy()
    nodes = nodes[0:min(args.subgraph_nodes_max_num, len(nodes))]
    nodes = list(set(nodes.tolist() + [i]))
    g_sub = dgl.node_subgraph(g, nodes)
    g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
    return g_sub



#此子图采样太耗时,已废弃
# #采样子图函数
# def get_subgraph(i, g, args):
#     #采样第i个节点的子图
#     start = time.time()
#     k = args.k_hop
#     g_sub, _ = dgl.khop_in_subgraph(g, i, k)
#     g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
#     end1 = time.time()
#     print("前置子图采样过程耗时:", end1 - start)
#     #防止采样的子图规模过小
#     while g_sub.number_of_nodes() < args.subgraph_nodes_min_num and g_sub.number_of_edges() < args.subgraph_edges_min_num:
#         k = k + 1
#         g_sub, _ = dgl.khop_in_subgraph(g, i, k)
#         g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
#         if k > 3:
#             break
#     end2 = time.time()
#     print("当节点数目过小时重新子图采样过程耗时:", end2 - end1)
#     #防止子图的节点数目过大
#     while g_sub.number_of_nodes() > args.subgraph_nodes_max_num:
#         if k > 1:
#             k = k -1
#             g_sub, _ = dgl.khop_in_subgraph(g, i, k)
#             g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
#         else:
#             #k降到1后子图的规模仍然过大，变为邻域子图采样
#             nodes = g.successors(i).numpy()
#             indices = tf.argsort(g.in_degrees(nodes.tolist()), direction='DESCENDING')[0:args.subgraph_nodes_max_num]
#             nodes = list(set(nodes[indices].tolist() + [i]))
#             g_sub = dgl.node_subgraph(g, nodes)
#             g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
#             break
#     end3 = time.time()
#     print("防止节点数目过大时重新子图采样过程耗时:", end3 - end2)
#     #防止采样的子图边的数目过大
#     if g_sub.number_of_edges() / args.subgraph_edges_max_num >= 1.2:
#         index = int(g_sub.number_of_nodes() / 5) + 1
#     elif g_sub.number_of_edges() > args.subgraph_edges_max_num and g_sub.number_of_edges() / \
#             args.subgraph_edges_max_num < 1.2:
#         index = int(g_sub.number_of_nodes() / 3) + 1
#     else:
#         index = 1
#     if g_sub.number_of_edges() > args.subgraph_edges_max_num:
#         nodes = g.successors(i).numpy()
#         indices = tf.argsort(g.in_degrees(nodes.tolist()), direction='DESCENDING').numpy()
#     s = 0
#     end4 = time.time()
#     print("防止采样的子图边的数目过大前置过程耗时:", end4 - end3)
#     while g_sub.number_of_edges() > args.subgraph_edges_max_num:
#         if k > 1:
#             k = k - 1
#             g_sub, _ = dgl.khop_in_subgraph(g, i, k)
#             g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
#         else:
#             #k降到1后子图的规模仍然过大,此时变为邻域子图采样
#             s = s + 2
#             indices1 = indices[:-min(index, len(indices) - 1)]
#
#             nodes1 = list(set(nodes[indices1].tolist() + [i]))
#             g_sub = dgl.node_subgraph(g, nodes1)
#             g_sub = dgl.to_bidirected(g_sub, copy_ndata=True)
#             index = index + s
#             if index > len(indices) - 1:
#                 break
#     end5 = time.time()
#     print("节点数目满足要求,但边数过大重新采样过程耗时:", end5 - end4)
#     return g_sub