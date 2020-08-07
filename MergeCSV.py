#!/usr/bin/python3
# -*- coding: utf-8 -*-


from AddLabel import *                                 # 从本项目的其他python文件中导入函数、类、接口


path = r"D:\TestFile\douyinwithlabel.csv"
merge_path = r"D:\TestFile\merge.csv"

def merge_csv(path1, path2):
    '''
    :param path1:
    :param path2:
    :expanlanation: 将path1和path2的数据合并，并存储在path1
    '''
    data1 = read_data_from_csv(path1)
    data2 = read_data_from_csv(path2)
    data1 = np.r_[data1, data2]
    save_data_to_csv(path1, data1)

merge_csv(merge_path, path);
