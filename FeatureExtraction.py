#!/usr/bin/python3
# -*- coding: utf-8 -*-

from AddLabel import *
from sklearn.decomposition import PCA
from DataProcessing import *


win_interval_size = 0.32                                     # window_interval大小32ms为一个时间间隔
win_interval_stride = 0.16                                   # 帧步幅 每间隔16ms取下一帧
NFFT = 512                                                   # 傅里叶变换所用参数
sample_rate = 50
win_size = 2
win_stride = 0.5
FEATURE = 12
LABEL = 2                                                   # douyin的标签为0，taobao的标签是1，kugou的标签是2，zhihu的标签是3



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


def computatef(mag_array):  # 计算得到特征矩阵，rd是ndarray类型

    # 计算特征
    win_max = np.max(mag_array)  # 计算窗口内磁力计最大值,  l---length
    win_min = np.min(mag_array)  # 计算窗口内磁力计最小值
    win_mean = np.mean(mag_array)  # 计算窗口内磁力计平均长度
    win_var = np.var(mag_array)  # 计算窗口内磁力计长度方差

        # 计算包长度峰度
    if win_var == 0:
        win_kurt = 0
    else:
        win_kurt = np.mean((mag_array - win_mean) ** 4)/pow(win_var, 2)  # 计算包长度峰度
        win_skew = np.mean((mag_array - win_mean) ** 3)   # 计算包长度偏斜度
        # win_sum = np.sum(mag_array)   # 计算窗口内总的包长度

        # 计算3个特殊位置，窗口大小的1/4把窗口分为两个部分，对两个部分分别计算包长度方差作为两个特征，同理，2/4处，3/4处，可得6个特征
    if mag_array.shape[0] < 4:  # 判断窗口大小是否有3个位置
        win1_4fmvar = 0
        win1_4lavar = 0
        win2_4fmvar = 0
        win2_4lavar = 0
        win3_4fmvar = 0
        win3_4lavar = 0
    else:  # 窗口大小有3个位置
        t = mag_array.shape[0]//4
        win1_4fm = mag_array[:t]  # 1/4的前边部分， fm---former,  pl---packet length
        win1_4la = mag_array[t:]  # 1/4的后边部分,  la---latter
        win2_4fm = mag_array[:t*2]
        win2_4la = mag_array[t*2:]
        win3_4fm = mag_array[:t*3]
        win3_4la = mag_array[t*3:]
        win1_4fmvar = np.var(win1_4fm)  # 计算1/4前边部分的方差
        win1_4lavar = np.var(win1_4la)  # 计算1/4后边部分的方差
        win2_4fmvar = np.var(win2_4fm)
        win2_4lavar = np.var(win2_4la)
        win3_4fmvar = np.var(win3_4fm)
        win3_4lavar = np.var(win3_4la)

    fvlm = np.asmatrix([win_max, win_min, win_mean, win_var, win_kurt, win_skew, win1_4fmvar, win1_4lavar, win2_4fmvar,
                                win2_4lavar, win3_4fmvar, win3_4lavar])  # 将该窗口内的特征值组合为矩阵
            # 如果是第一个窗口，则赋值给fv，否，则在fv后合并下一个窗口特征矩阵
    return fvlm


path1 = r"C:\Users\14167\Desktop\20200924\Test\VoiceCall\merge.csv"
csv_data = read_data_from_csv(path1)
x_axis = csv_data[:, 0]
y_axis = csv_data[:, 1]
z_axis = csv_data[:, 2]

# mag_value = csv_data[:, 3]                             # 如果在该文件前统一对所有的数据进行预处理

win_len = win_size * sample_rate
win_step = win_stride * sample_rate
win_len = int(round(win_len))
win_step = int(round(win_step))
num_len = x_axis.shape[0]
num_win = 1 + int(np.ceil(float(np.abs(num_len - win_len)) / win_step))
pad_len = num_win * win_step + win_len                                       # 反过来计算分帧后的一维数组数据总数
z = np.zeros((pad_len - num_len))                                            # 多余的数据用0填充，生成一个全0元素矩阵
pad_x = np.append(x_axis, z)
pad_y = np.append(y_axis, z)
pad_z = np.append(z_axis, z)
# feature_vector = np.zeros((12*num_win, FEATURE))                             # 12是num_win_interval的值，也就是一个窗口划分的interval的数量
fv = np.zeros((num_win, 12))
win = []

'''
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
'''

for i in range(num_win):
    if i == 0:
        win_x = pad_x[:win_len]
        win_y = pad_y[:win_len]
        win_z = pad_z[:win_len]
    else:
        win_x = pad_x[i*win_step:i*win_step+win_len]
        win_y = pad_y[i * win_step:i * win_step + win_len]
        win_z = pad_z[i * win_step:i * win_step + win_len]

    centralization(win_x)
    centralization(win_y)
    centralization(win_z)

    win_mag_aggr = aggregation(win_x, win_y, win_z)
    win_mag_norm = normalization(win_mag_aggr)

    mag_array = win_mag_norm.reshape((-1, 1))  # 将一维数组转换成2维矩阵得到包长度矩阵m*1，  l---length
    fv[i] = computatef(mag_array)

path2 = r"C:\Users\14167\Desktop\20200924\Test\VoiceCall\withfeatureandlabel.csv"
feature_vector = add_label(fv, LABEL)
save_data_with_label_to_csv(path2, feature_vector)












