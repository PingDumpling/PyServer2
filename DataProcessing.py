#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from AddLabel import *


def centralization(axis_data):
    '''
    :param axis_data: 收集到的每一个轴上的原始数据
    :return: 中心化后的每一个轴上的数据
    '''
    mean = np.mean(axis_data)
    for i in range(axis_data.shape[0]):                       # axis_data.shape[0]取得矩阵的行数
        axis_data[i] -= mean


def aggregation(x, y, z):
    '''
    :param x: 中心化后x轴的数据
    :param y: 中心化后y轴的数据
    :param z: 中心化后z轴的数据
    :return: 合磁场大小
    '''
    mag_aggr = np.zeros((x.shape[0], 1))                        # 用零初始化(x.shape[0],1)矩阵
    for i in range(x.shape[0]):
        mag_aggr[i] = math.sqrt(math.pow(x[i], 2) + math.pow(y[i], 2) + math.pow(z[i], 2))
    return mag_aggr


def normalization(mag_aggr):
    '''
    :param aggr_data: aggregation后的合磁力计读数
    :return: 归一化后的合磁力计读数
    '''
    mag_norm = np.zeros((mag_aggr.shape[0], 1))
    for i in range(mag_aggr.shape[0]):
        mag_norm[i] = (mag_aggr[i] - np.min(mag_aggr))/(np.max(mag_aggr) - np.min(mag_aggr))
    return mag_norm


path1 = r"D:\TestFile\douyin.csv"
path2 = r"D:\TestFile\afterdataprocessing_douyin.csv"
raw_data = read_data_from_csv(path1)
raw_x = raw_data[:, 0]                                          # 把x轴的值取出来
centralization(raw_x)
raw_y = raw_data[:, 1]
centralization(raw_y)
raw_z = raw_data[:, 2]
centralization(raw_z)

aggr_mag_value = aggregation(raw_x, raw_y, raw_z)
# print("归一化前：")
# print(aggr_mag_value[:3])                                     # 输出前3行的数据
norm_mag_value = normalization(aggr_mag_value)
mag_aggr = np.zeros((raw_data.shape[0], raw_data.shape[1]))
mag_aggr[:, 0] = raw_x
mag_aggr[:, 1] = raw_y
mag_aggr[:, 2] = raw_z
mag_aggr[:, 3] = norm_mag_value[:, 0]

save_data_to_csv(path2, mag_aggr)


