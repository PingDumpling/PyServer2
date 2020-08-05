#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

path1 = r'D:\TestFile\Beforetest.csv'
path2 = r'D:\TestFile\AfterDeleteWeChatText.csv'


def del_same_mag_value(bef_data):
    (row, len) = bef_data.shape                                  # 得到bef_data的行和列
    aft_data = np.array([[0.0]*len for i in range(row)])         # 创建一个全值为0.0的二维数组row*len
    str_data = [' ']*row                                         # 创建一个全值为' '的字符串数组
    for i in range(row):
        str_data[i] = str(bef_data[i][3])                        # 将合磁场强度转换成字符串赋值给str_data
    j = 0                                                        # j为删除重复值后的二维数组的索引
    for i in range(row):
        if i+1 == row:                                           # 避免数组索引越界
            break
        else:
            if str_data[i] != str_data[i+1]:
                aft_data[j] = bef_data[i]
                j += 1

    aft_data[j] = bef_data[row-1]                                 # 将未删除前的数组的最后一行值赋值给删除后的数组
    aft_data = aft_data[:j + 1]                                   # 创建aft_data时是row*len,而我们只需要j+1*len,后面的数据全是0.0
    return aft_data


before_data = pd.read_csv(path1).values
after_data = np.array(del_same_mag_value(before_data))
after_data = pd.DataFrame(after_data)
after_data.to_csv(path2, header=['X', 'Y', 'Z', 'Aggre', 'Label'], index=False)

