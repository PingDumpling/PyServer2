#!/usr/bin/python3
# -*- coding: utf-8 -*-

import socket
import sys
from time import ctime


# 1.socket(socket_family, socket_type, protocol=0)
# 其中，socket_family 是 AF_UNIX 或 AF_INET,ocket_type 是 SOCK_STREAM或 SOCK_DGRAM, protocol 通常省略，默认为 0。
# 为了创建 TCP/IP 套接字，可以用下面的方式调用 socket.socket()。
# tcpSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 同样，为了创建 UDP/IP 套接字，需要执行以下语句。
# udpSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 本地机器名
host = "113.54.211.59"   # 真机测试
# host = "127.0.0.1"    模拟器测试


# 设置端口
port = 5555

# 设置最大缓冲区长度
BUFSIZE = 4096

# 2.s.bind绑定本地地址到socket对象
ServerSocket.bind((host, port))
# 3.s.listen监听地址端口，连接几个客户端
ServerSocket.listen(5)
while True:
    # 4.s.accept阻塞接受链接请求，被动接受 TCP 客户端连接，一直等待直到连接到达（阻塞）
    # accept()方法会返回一个含有两个元素的元组（fd,addr）。
    # 第一个元素是新的socket对象，服务器通过它与客户端通信。
    # 第二个元素也是元组，是客户端的地址及端口信息。
    print("等待连接：")
    clientsocket, addr = ServerSocket.accept()
    print("连接地址:%s" % str(addr))
    while True:
        msg = "向客户端发送的本条消息".encode("utf-8")
        # send()和recv()的数据格式都是bytes。
        # (str和bytes的相互转化，用encode()和decode(),或者用bytes()和str())
        clientsocket.send(msg)
        data = clientsocket.recv(BUFSIZE).decode("utf-8")
        print(data)
        if not data:
            break
        data1 = ('[%s] %s' % (ctime(), data)).encode("utf-8")
        clientsocket.send(data1)
        # clientsocket.settimeout(20)

    clientsocket.close()

ServerSocket.close()

