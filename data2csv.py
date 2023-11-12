#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 9:27
# @Author  : liangliang
# @File    : data2csv.py
# @Software: PyCharm

import argparse
import time
import pandas as pd
import os
import datetime

def is_list(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return True
    else:
        return False

def write(data, count):
    # 注意在此业务中data是一个二维list
    # 数据的数量
    print("开始写入第{}个文件数据. {}".format(count, datetime.datetime.now()))
    n = len(data)
    flag = is_list(data[0])
    if n > 0:
        start = time.time()
        with open(os.path.join(args.data_output, 'preds_{}.csv'.format(count)), mode="a") as resultfile:
            if flag:
                for i in range(n):
                    line = ",".join(map(str, data[i])) + "\n"
                    resultfile.write(line)
            else:
                line = ",".join(map(str, data)) + "\n"
                resultfile.write(line)
        cost = time.time() - start
        print("第{}个大数据文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(count, n, cost, datetime.datetime.now()))

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    input_path = args.data_input.split(",")[0]
    n = 0
    input_files = []
    for _, _, files in os.walk(input_path):
        input_files = input_files + sorted([file for file in files if file.find("pred") != -1])
    n = len(input_files)
    print("一共有{}个文件. {}".format(n, datetime.datetime.now()))
    for i in range(n):
        data = pd.read_csv(os.path.join(input_path, "pred_{}.csv".format(i)), sep=',', header=None)
        res = ""
        s = data.iloc[0]
        s_len = len(s)
        s = ",".join(s)
        s = s.replace('[', '')
        s = s.replace('],', ';')
        s = s.replace(']', '')
        s = s.replace("'", "")
        res = s.split(';')
        result = []
        for item in res:
            item = item.split(',')
            result.append(list(item))
        write(result, i)
