#!/usr/bin/python3
# -*- coding: utf-8 -*-

import socket
import sys
from time import ctime
import pandas as pd
from decimal import Decimal
import numpy as np

# 1.socket(socket_family, socket_type, protocol=0)
# 其中，socket_family 是 AF_UNIX 或 AF_INET,ocket_type 是 SOCK_STREAM或 SOCK_DGRAM, protocol 通常省略，默认为 0。
# 为了创建 TCP/IP 套接字，可以用下面的方式调用 socket.socket()。
# tcpSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 同样，为了创建 UDP/IP 套接字，需要执行以下语句。
# udpSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
from numpy import matrix

ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 本地机器名
host = "113.54.211.59"   # 真机测试
# host = "127.0.0.1"    模拟器测试


# 设置端口
port = 8765

# 设置最大缓冲区长度
BUFSIZE = 1024

# 传输的数据是否带有合磁场强度,4表示有，3表示只有X轴、Y轴、Z轴
TRANDATALEN = 4

'''
功能：将从Android客户端读取到的数据分割成X轴、Y轴、Z轴、合磁场
@:param str 从Android客户端读取到的数据，是字符串形式，以','分隔
@:param mag_value 分割后的数据，类型是matrix (4*1)
'''


def split_string(str, mag_value):
    tem_data1 = str.split(",")
    tem_data2 = tem_data1[:len(tem_data1)-1]
    print("tem_data2")
    print(tem_data2)
    for i in range(len(tem_data2)):
        # tem_data2[i] = float(Decimal(tem_data2[i]).quantize(Decimal('0.000')))     # 保留小数点后3位，并将Decimal类型强制转换成float类型
        tem_data2[i] = float(tem_data2[i])        # 将String类型强制转换成float类型

    tem_data2 = np.mat(tem_data2)                   # 将list类型转换成matrix
    row = int(tem_data2.shape[1]/TRANDATALEN)
    mag_value = np.zeros((row, TRANDATALEN))
    for i in range(row):
        mag_value[i] = tem_data2[:, i * TRANDATALEN:i * TRANDATALEN + TRANDATALEN]
    return mag_value


ServerSocket.bind((host, port))    # 2.s.bind绑定本地地址到socket对象
ServerSocket.listen(5)             # 3.s.listen监听地址端口，连接几个客户端

while True:
    # 4.s.accept阻塞接受链接请求，被动接受 TCP 客户端连接，一直等待直到连接到达（阻塞）
    # accept()方法会返回一个含有两个元素的元组（fd,addr）。
    # 第一个元素是新的socket对象，服务器通过它与客户端通信。
    # 第二个元素也是元组，是客户端的地址及端口信息。
    print("等待连接：")
    clientsocket, addr = ServerSocket.accept()
    print("连接地址:%s" % str(addr))

    flag1 = 1                      # 标志位，看数据保存文件csv是否第一次打开
    while True:
        flag2 = 1                  # 标志位，用以标志clientsocket是否被关闭
        msg = "向客户端发送的本条消息".encode("utf-8")            # send()和recv()的数据格式都是bytes。
        # clientsocket.send(msg)                                  # (str和bytes的相互转化，用encode()和decode(),或者用bytes()和str())
        data = clientsocket.recv(BUFSIZE).decode("utf-8")
        if not data:                    # 判断缓冲区是否还有数据，若没有，就关闭socket连接
            break
            flag2 = 0
            clientsocket.close()

        mag_value = 0                   # 数组(4*1) ----保存X,Y,Z轴，合磁场强度
        mag_value = split_string(data, mag_value)
        mag_value = pd.DataFrame(mag_value)
        path = r'D:\TestFile\test.csv'
        if flag1 == 1:
            flag1 = 0
            mag_value.to_csv(path, header=['X', 'Y', 'Z', 'Aggre'], index=False)    # 带有Aggre表示合磁场强度
            # mag_value.to_csv(path, header=['X', 'Y', 'Z'], index=False)      # 只有X轴、Y轴、Z轴，没有合磁场强度
        else:
            mag_value.to_csv(path, header=False, index=False, mode='a+')


        # print(type(data))
        print(data)

        # print(mag_value)
        # data1 = ('[%s]' % (ctime())).encode("utf-8")
        # clientsocket.send(data1)
        # clientsocket.settimeout(20)
    if flag2 == 1:
        clientsocket.close()

ServerSocket.close()








