#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# 采样频率,F的值表示1s多少个点
F = 50

# 数据的个数
NUM = 300


def compute_x_axis_time(NUM):
    '''
    :param mag_value: ndarray
    :return: time: list
    :explanation: 计算横坐标
    '''

    # row = mag_value.shape[0]
    time = [0.0]*NUM
    interval = 1.0/F*1000                                   # 时间间隔,乘1000是为了将横坐标从s转换为ms
    for i in range(NUM):
        if i+1 != NUM:
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

    plt.plot(time, mag_value)                               # 横坐标为time,纵坐标为mag_value,二者类型必须为list类型
    plt.xlabel("time(ms)")
    plt.ylabel("合磁场大小")
    plt.title("合磁场大小随着时间的变化曲线图")
    # plt.savefig('MagValueWeChatVoice.png')                # savefig()必须在show()之前，不然保存的就是空的图
    plt.show()


def plot_two_subplot(mag_15, mag_16, time):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlabel("time(ms)")
    plt.ylabel("Text_15号的合磁场大小")
    plt.title("Text_15号磁场随着时间的变化曲线图")
    # plt.savefig('MagValueWeChatVoice.png')                # savefig()必须在show()之前，不然保存的就是空的图
    plt.plot(time, mag_15)
    # plt.show()
    plt.subplot(2, 1, 2)
    plt.xlabel("time(ms)")
    plt.ylabel("Text_16号的合磁场大小")
    plt.title("Text_16号磁场随着时间的变化曲线图")
    # plt.savefig('MagValueWeChatVoice.png')                # savefig()必须在show()之前，不然保存的就是空的图
    plt.plot(time, mag_16)
    plt.show()


def plot_three_subplot(mag_t, mag_v, mag_vc, time):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.xlabel("time(ms)")
    plt.ylabel("Text_15号的合磁场大小")
    plt.title("Text_15号磁场随着时间的变化曲线图")
    plt.plot(time, mag_t)
    plt.subplot(3, 1, 2)
    plt.xlabel("time(ms)")
    plt.ylabel("Voice_15号的合磁场大小")
    plt.title("Voice_15号磁场随着时间的变化曲线图")
    plt.plot(time, mag_v)
    plt.subplot(3, 1, 3)
    plt.xlabel("time(ms)")
    plt.ylabel("VoiceCall_15号的合磁场大小")
    plt.title("VoiceCall_15号磁场随着时间的变化曲线图")
    plt.plot(time, mag_vc)
    plt.show()


'''
path1 = r'D:\TestFile\WeChat\20200915\Text\afterdataprocessing.csv'
path2 = r'D:\TestFile\WeChat\20200916\Text\afterdataprocessing.csv'

read_data_15 = pd.read_csv(path1).values
mag_value_15 = read_data_15[:, 3]                                 # 获取合磁场强度列
mag_value_15 = mag_value_15[0:NUM]
time = compute_x_axis_time(NUM)
mag_value_15 = mag_value_15.tolist()                              # 将ndarray类型转换成list类型

read_data_16 = pd.read_csv(path2).values
mag_value_16 = read_data_16[:, 3]                                 # 获取合磁场强度列
mag_value_16 = mag_value_16[0:NUM]
time = compute_x_axis_time(NUM)
mag_value_16 = mag_value_16.tolist()                              # 将ndarray类型转换成list类型
plot_two_subplot(mag_value_15, mag_value_16, time)
'''


path1 = r'D:\TestFile\WeChat\20200916\Text\afterdataprocessing.csv'
path2 = r'D:\TestFile\WeChat\20200916\Voice\afterdataprocessing.csv'
path3 = r'D:\TestFile\WeChat\20200916\VoiceCall\afterdataprocessing.csv'

read_data_t = pd.read_csv(path1).values
mag_value_t = read_data_t[:, 3]                                 # 获取合磁场强度列
mag_value_t = mag_value_t[0:NUM]
time = compute_x_axis_time(NUM)
mag_value_t = mag_value_t.tolist()                              # 将ndarray类型转换成list类型

read_data_v = pd.read_csv(path2).values
mag_value_v = read_data_v[:, 3]                                 # 获取合磁场强度列
mag_value_v = mag_value_v[0:NUM]
time = compute_x_axis_time(NUM)
mag_value_v = mag_value_v.tolist()                              # 将ndarray类型转换成list类型

read_data_vc = pd.read_csv(path3).values
mag_value_vc = read_data_vc[:, 3]                                 # 获取合磁场强度列
mag_value_vc = mag_value_vc[0:NUM]
time = compute_x_axis_time(NUM)
mag_value_vc = mag_value_vc.tolist()                              # 将ndarray类型转换成list类型
plot_three_subplot(mag_value_t, mag_value_v, mag_value_vc, time)


