#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Compatibility layer between Python 2 and Python 3
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization


# %%


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def show_confusion_matrix(validations, predictions):

    """
    该函数的功能是将混淆矩阵显示为热力图
    :param validations: 真实标记
    :param predictions: 预测标记
    :return:
    """
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))                # figure的大小位宽6inch,长4inch
    sns.heatmap(matrix,                       # 将混淆矩阵显示为热力图
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def show_basic_dataframe_info(dataframe,
                              preview_rows=20):
    """
    This function shows basic information for the given dataframe
    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview
    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe())


def read_data(file_path):
    """
    This function reads the accelerometer data from a file
    Args:
        file_path: URL pointing to the CSV file
    Returns:
        A pandas dataframe
    """

    column_names = ['x-axis',
                    'y-axis',
                    'z-axis',
                    'label']
    df = pd.read_csv(file_path)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    # df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except():
        return np.nan


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])               # 新建刻度，y轴的
    ax.set_xlim([min(x), max(x)])                                       # 新建刻度，x轴的范围
    ax.grid(True)

def compute_x_axis_time(NUM):
    '''
    :param mag_value: ndarray
    :return: time: list
    :explanation: 计算横坐标
    '''
    F = 50
    # row = mag_value.shape[0]
    time = [0.0]*NUM
    interval = 1.0/F*1000                                   # 时间间隔,乘1000是为了将横坐标从s转换为ms
    for i in range(NUM):
        if i+1 != NUM:
            time[i+1] = time[i] + interval

    return time
'''
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,            # 建立3个子图，以行排列，ax0表示得到第一个图的坐标轴
                                        figsize=(15, 10),
                                        sharex=True)
    time = compute_x_axis_time(len(data['x-axis']))
    plot_axis(ax0, time, data['x-axis'], 'x-axis')
    plot_axis(ax1, time, data['y-axis'], 'y-axis')
    plot_axis(ax2, time, data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
'''


def create_segments_and_labels(df, time_steps, step):
    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    # labels = set()
    num_time_periods = 0
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + time_steps])[0][0]       # mode()[0][0]为返回矩阵或者数组中最常出现的成员，
                                                                       # mode()[1][0]为最常出现的成员出现的次数
        # label = df['label'].values[i: i + time_steps].ravel()

        segments.append([xs, ys, zs])
        num_time_periods += 1
        labels.append(label)
        # labels.add(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(num_time_periods, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels, num_time_periods


# %%

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

LABELS = ["0",
          "1",
          "2",
          ]

# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40

# %%

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df_train = read_data(r"C:\Users\14167\Desktop\CNNdata\train\merge.csv")
df_test = read_data(r"C:\Users\14167\Desktop\CNNdata\test\merge.csv")

# Describe the data
show_basic_dataframe_info(df_train, 20)


# Define column name of the label vector
LABEL = "ServiceEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()                       # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内，并且根据字典排序
# Add a new column to the existing DataFrame with the encoded values
df_train[LABEL] = le.fit_transform(df_train["label"].values.ravel())        # ravel()是将数组维度拉成1维数组

# %%

print("\n--- Reshape the data into segments ---\n")

# Normalize features for training data set
df_train['x-axis'] = feature_normalize(df_train['x-axis'])
df_train['y-axis'] = feature_normalize(df_train['y-axis'])
df_train['z-axis'] = feature_normalize(df_train['z-axis'])
# Round in order to comply to NSNumber from iOS
df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})              # round()函数指定列保留6位小数

# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train, num_time_periods = create_segments_and_labels(df_train, TIME_PERIODS, STEP_DISTANCE)


# %%

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (20869, 80, 3)
print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train shape: ', y_train.shape)
# Displays (20869,)

# Set input & output dimensions
TIME_PERIODS, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (TIME_PERIODS * num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)
# x_train shape: (20869, 240)
print('input_shape:', input_shape)
# input_shape: (240)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

x_train = x_train.reshape(num_time_periods, TIME_PERIODS, 3, order='F')

# %%

# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%

print("\n--- Create neural network model ---\n")

# 1D CNN neural network

model_m = Sequential()
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
#model_m.add(BatchNormalization(input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
# model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())

# Accuracy on training data: 99%
# Accuracy on test data: 91%

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,training stops early
'''
# 使用回调函数来观察训练过程中网络内部的状态和统计信息
# ModelCheckpoint存储最优的模型
# filepath为我们存储的位置和模型名称，以.h5为后缀，
# monitor为检测的指标，这里我们检测验证集里面的误差率，
# save_best_only代表我们只保存最优的训练结果。
# validation_data就是给定的验证集数据。
# val_loss:验证集的误差
'''
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),


    # patience当连续多少个epochs时验证集精度不再变好从而终止训练，这里选择了1
    # monitor为选择的检测指标，这里选择检测’accuracy’识别率为指标，
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]

'''
    # 学习率动态调整
    # 1.monitor：被监测的量
    # 2. factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    # 3. patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    # 4. mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    # 5. epsilon：阈值，用来确定是否进入检测值的“平原区”
    # 6. cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    # 7. min_lr：学习率的下限

    # 当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果
    keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.1, 
                patience=10, 
                verbose=0, 
                mode='auto', 
                epsilon=0.0001, 
                cooldown=0, 
                min_lr=0
                            )
    '''


model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])                         # metrics=['accuracy']有验证集的正确率和误差时需要的参数

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 10

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,                # mini-batch梯度下降
                      epochs=EPOCHS,                        # epochs指的就是训练过程中数据将被“轮”多少次
                      callbacks=callbacks_list,
                      validation_split=0.2,                 # 划分验证集
                      verbose=1)                            # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
# plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# %%

print("\n--- Check against test data ---\n")

# Normalize features for training data set
df_test['x-axis'] = feature_normalize(df_test['x-axis'])
df_test['y-axis'] = feature_normalize(df_test['y-axis'])
df_test['z-axis'] = feature_normalize(df_test['z-axis'])

df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test, num_time_periods = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            )

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

x_test = x_test.reshape(num_time_periods, TIME_PERIODS, 3, order='F')           # 按列重构数组

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

# %%

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))
