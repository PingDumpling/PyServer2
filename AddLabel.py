#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


path1 = r"D:\TestFile\douyin.csv"
path2 = r"D:\TestFile\douyinwithlabel.csv"
Label = 3


def read_data_from_csv(path):
    '''
    :param path: 路径
    :return: data: ndarray,从csv中读取的数据
    '''
    data = pd.read_csv(path).values
    return data


def save_data_with_label_to_csv(path, data):
    '''
    :param path: 保存文件路径
    :param data: ndarray,要保存的数据,带有标签
    :return:
    '''
    data = pd.DataFrame(data)
    data.to_csv(path, header=['X', 'Y', 'Z', 'Aggre', 'Label'], index=False)


def save_data_to_csv(path, data):
    '''
    :param path: 保存文件路径
    :param data: ndarray,要保存的数据,不带有标签
    :return:
    '''
    data = pd.DataFrame(data)
    data.to_csv(path, header=['X', 'Y', 'Z', 'Aggre'], index=False)


def add_label(data, y):
    '''
    :param data: 从csv中读取的原始数据
    :return: data: 加了标记后的数据
    '''
    label = []
    for i in range(data.shape[0]):
        label.append(y)                                # 给label赋值
    data = np.c_[data, label]                          # 在data后增加一列label
    return data


data = read_data_from_csv(path1)
data = add_label(data, Label)
save_data_with_label_to_csv(path2, data)





