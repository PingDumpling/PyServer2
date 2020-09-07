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
    for i in range(axis_data.shape[0]):
        axis_data[i] -= mean


def aggregation(x, y, z):
    '''
    :param x: 中心化后x轴的数据
    :param y: 中心化后y轴的数据
    :param z: 中心化后z轴的数据
    :return: 合磁场大小
    '''
    aggr_mag = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        aggr_mag[i] = math.sqrt(math.pow(x[i], 2) + math.pow(y[i], 2) + math.pow(z[i], 2))
    return aggr_mag


path = r"D:\TestFile\douyin.csv"
raw_data = read_data_from_csv(path)
raw_x = raw_data[:, 0]
centralization(raw_x)
raw_y = raw_data[:, 1]
centralization(raw_y)
raw_z = raw_data[:, 2]
centralization(raw_z)

aggr_mag_value = aggregation(raw_x, raw_y, raw_z)
print(aggr_mag_value[:3])

