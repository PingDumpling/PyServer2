#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

path = r'D:\TestFile\AfterDeleteWeChatText.csv'

# 采样频率,F的值表示1s多少个点
F = 8


def compute_x_axis_time(mag_value):
    '''
    :param mag_value: ndarray
    :return: time: list
    :explanation: 计算横坐标
    '''

    row = mag_value.shape[0]
    time = [0.0]*row
    interval = 1.0/F*1000                                   # 时间间隔,乘1000是为了将横坐标从s转换为ms
    for i in range(row):
        if i+1 != row:
            time[i+1] = time[i] + interval

    return time


def plot_mag_value_curve(mag_value, time):
    # pf = plt.figure()
    '''
       :param mag_value: list,  time: list
       :explanation: 画磁场随时间变化图
    '''

    '''
    这两行代码是解决xlabel和ylabel中中文显示乱码的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    '''

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(time, mag_value)                               # 横坐标为time,纵坐标为mag_value,二者类型必须为list类型
    plt.xlabel("time(ms)")
    plt.ylabel("合磁场大小")
    plt.title("合磁场大小随着时间的变化曲线图")
    # plt.savefig('MagValueWeChatVoice.png')                # savefig()必须在show()之前，不然保存的就是空的图
    plt.show()


read_data = pd.read_csv(path).values
mag_value = read_data[:, 3]                                 # 获取合磁场强度列

time = compute_x_axis_time(mag_value)
mag_value = mag_value.tolist()                              # 将ndarray类型转换成list类型

plot_mag_value_curve(mag_value, time)
