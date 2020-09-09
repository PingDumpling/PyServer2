#!/usr/bin/python3
# -*- coding: utf-8 -*-

from AddLabel import *
from sklearn.decomposition import PCA


win_interval_size = 0.32                                     # window_interval大小32ms为一个时间间隔
win_interval_stride = 0.16                                   # 帧步幅 每间隔16ms取下一帧
NFFT = 512                                                   # 傅里叶变换所用参数
sample_rate = 50
win_size = 2
win_stride = 1
FEATURE = 12
LABEL = 3                                                   # douyin的标签为0，taobao的标签是1，kugou的标签是2，zhihu的标签是3


def divide_win_interval(mag_aggr):
    '''
    :param mag_aggr: 从csv文件中读取的mag_aggr数据
    :return: 分时间间隔后的mag_matrix
    '''
    win_interval_length, win_interval_step = win_interval_size * sample_rate, win_interval_stride * sample_rate          # frame_length:表示一个帧有多少个采样点,frame_step:表示帧位移有多少个采样点

    win_interval_length = int(round(win_interval_length))
    win_interval_step = int(round(win_interval_step))
    num_len = mag_aggr.shape[0]
    num_win_interval = 1 + int(np.ceil(float(np.abs(num_len - win_interval_length)) / win_interval_step))
    pad_signal_length = num_win_interval * win_interval_step + win_interval_length                           # 反过来计算分帧后的一维数组数据总数
    z = np.zeros((pad_signal_length - num_len))                                            # 多余的数据用0填充，生成一个全0元素矩阵
    pad_signal = np.append(mag_aggr, z)
    indices = np.tile(np.arange(0, win_interval_length), (num_win_interval, 1)) + np.tile(
        np.arange(0, num_win_interval * win_interval_step, win_interval_step), (win_interval_length, 1)).T
    mag_matrix = pad_signal[np.mat(indices).astype(np.int32, copy=False)]                      # frames为num_frames*frame_length的二维矩
                                                                                               # 帧的数量
    return mag_matrix


# 加汉明窗
def add_window(mag_matrix, win_interval_length):
    mag_matrix *= np.hamming(win_interval_length)                            # frames还是为num_frames*frame_length的二维矩阵


# 进行傅里叶变换
def fft_trans(mag_matrix):
    '''
    :param mag_matrix: 经过划分时间间隔和加汉明窗后的mag_matrix
    :return: 经过fft变换后的到的频谱
    '''
    mag_fft = np.absolute(np.fft.rfft(mag_matrix, NFFT))                    # Magnitude of the FFT
    mag_pow = ((1.0 / NFFT) * (mag_fft ** 2))                               # Power Spectrum
    return mag_pow


def pca_bulid_feature(x):
    '''
    :param x: fft后的频谱结果
    :return: pca后的降维数据x
    '''
    pca =PCA(n_components=FEATURE)
    reduced_x = pca.fit_transform(x)
    return reduced_x


path1 = r"D:\TestFile\afterdataprocessing_douyin.csv"
csv_data = read_data_from_csv(path1)
mag_value = csv_data[:, 3]

win_len = win_size * sample_rate
win_step = win_stride * sample_rate
win_len = int(round(win_len))
win_step = int(round(win_step))
num_len = mag_value.shape[0]
num_win = 1 + int(np.ceil(float(np.abs(num_len - win_len)) / win_step))
pad_len = num_win * win_step + win_len                                       # 反过来计算分帧后的一维数组数据总数
z = np.zeros((pad_len - num_len))                                            # 多余的数据用0填充，生成一个全0元素矩阵
pad_mag = np.append(mag_value, z)
feature_vector = np.zeros((12*num_win, FEATURE))                             # 12是num_win_interval的值，也就是一个窗口划分的interval的数量
win = []

for i in range(num_win):
    if i == 0:
        win = pad_mag[:win_len-1]
    else:
        win = pad_mag[i*win_step:i*win_step+win_len-1]
    mag_matrix = divide_win_interval(win)
    add_window(mag_matrix, win_interval_size * sample_rate)
    mag_pow = fft_trans(mag_matrix)
    pca_mag = pca_bulid_feature(mag_pow)
    feature_vector[i*12:(i+1)*12] = pca_mag

path2 = r"D:\TestFile\douyinwithfeatureandlabel.csv"
feature_vector = add_label(feature_vector, LABEL)
save_data_with_label_to_csv(path2, feature_vector)











