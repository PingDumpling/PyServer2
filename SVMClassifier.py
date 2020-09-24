from AddLabel import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
from ComputeConfusionMatrix import *


# clf = SVC()

def accuracy(acc):
    print('测试集的accuracy为:')
    # 五折交叉验证，输出每组，一共5个accuracy，且保留到小数点后4位
    for i in range(len(acc)):
        print('%.4f' % acc[i], end=' ')
    print()
    # 输出acc的平均值
    print('测试集的accuracy平均值为：%.4f' % np.mean(acc))
    print()


def precision(pre):
    print('测试集的precision为:')
    for i in range(len(pre)):
        print('%.4f' % pre[i], end=' ')
    print()
    print('测试集的precision平均值为：%.4f' % np.mean(pre))
    print()


def recall(rec):
    print('测试集的recall为:')
    for i in range(len(rec)):
        print('%.4f' % rec[i], end=' ')
    print()
    print('测试集的recall平均值为：%.4f' % np.mean(rec))
    print()


def f1score(f1_sco):
    print('测试集的f1_score为:')
    for i in range(len(f1_sco)):
        print('%.4f' % f1_sco[i], end=' ')
    print()
    print('测试集的f1_score平均值为：%.4f' % np.mean(f1_sco))
    print()


def predict_to_csv(path, y_test, y_hat):
    data = np.c_[y_test, y_hat]
    data = pd.DataFrame(data)
    data.to_csv(path, header=['y_test', 'y_hat'], index=False)

def ClfRF_cv(x, y):
    '''
    :param x: 训练的特征数据
    :param y: 训练的标签数据
    :return:
    explanation: 用x,y进行交叉验证一次来确定最好的模型超参数
    '''

    acc = []  # 初始化acc
    f1_sco = []
    pre = []
    rec = []
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in skfold.split(x, y):  # 对数据建立5折交叉验证的划分
        # for test_index,train_index in New_sam.split(Sam):  #默认第一个参数是训练集，第二个参数是测试集
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print('训练集数量:', x_train.shape, '测试集数量:', x_test.shape)  # 结果表明每次划分的数量
        clf = RandomForestClassifier(n_estimators=115, max_features='log2')  # 定义随机森林分类器， 树的个数为50
        clf.fit(x_train, y_train)  # 训练分类器
        y_hat = clf.predict(x_test)  # 预测测试集的类别
        cm = confusion_matrix(y_test, y_hat, labels=[0, 1, 2])  # 生成混淆矩阵 混淆矩阵的第一个值cm[0][0]表示实际为0类，预测为0类的样本数
        # cm2[1][0]表示实际为1类，预测为0类的样本数
        # print(classification_report(y_test, y_hat, target_names=["browsing", "text", "voice", "video"],digits=2))
        # 假设标签1代表browsing、标签2代表text、标签3代表voice、标签4代表video digits可以设置小数点后保留的位数 默认是2
        # 准确率 正确分类的样本数 比上 总样本数
        acc.append(accuracy_score(y_test, y_hat))  # 混淆矩阵对角线元素之和 比上 混淆矩阵所有元素和
        pre.append(precision_score(y_test, y_hat, average='macro'))  # 宏平均（Macro-averaging）是指所有类别的每一个统计指标值的算数平均值
        rec.append(recall_score(y_test, y_hat, average='macro'))
        f1_sco.append(f1_score(y_test, y_hat, average='macro'))

    accuracy(acc)
    precision(pre)
    recall(rec)
    f1score(f1_sco)

    # path = r'D:\TestFile\WeChat\compare_y_testandy_hat\voice_voicecall.csv'
    # predict_to_csv(path, y_test, y_hat)
    '''
    with open(r'D:\TestFile\WeChat\20200916\model\svm_text_voice_voicecall.pickle', 'wb') as fw:
        pickle.dump(clf, fw)
        print("done")
    '''


def clf_rf_split_train(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0, stratify=y)

    clf = RandomForestClassifier(n_estimators=115, max_features='log2')  # 定义随机森林分类器， 树的个数为50
    clf.fit(x_train, y_train)  # 训练分类器

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  # 生成混淆矩阵 混淆矩阵的第一个值cm[0][0]表示实际为0类，预测为0类的样本数
    # cm = comp_conf_mat(y_test, y_pred)
    # cm2[1][0]表示实际为1类，预测为0类的样本数
    # print(classification_report(y_test, y_hat, target_names=["browsing", "text", "voice", "video"],digits=2))
    # 假设标签1代表browsing、标签2代表text、标签3代表voice、标签4代表video digits可以设置小数点后保留的位数 默认是2
    # 准确率 正确分类的样本数 比上 总样本数
    print("测试集的accuracy为：")
    print(accuracy_score(y_test, y_pred))  # 混淆矩阵对角线元素之和 比上 混淆矩阵所有元素和
    print("测试集的precision为：")
    print(precision_score(y_test, y_pred, average='macro',
                          zero_division='warn'))  # 宏平均（Macro-averaging）是指所有类别的每一个统计指标值的算数平均值
    print("测试集的recall为：")
    print(recall_score(y_test, y_pred, average='macro', zero_division='warn'))
    print("测试集的f1_score为：")
    print(f1_score(y_test, y_pred, average='macro', zero_division='warn'))


def clf_rf_test(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=115, max_features='log2')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    # cm = comp_conf_mat(y_test, y_pred)
    print("混淆矩阵为：")
    print(cm)
    print("测试集的accuracy为：")
    print(accuracy_score(y_test, y_pred))  # 混淆矩阵对角线元素之和 比上 混淆矩阵所有元素和
    print("测试集的precision为：")
    print(precision_score(y_test, y_pred, average='macro',
                          zero_division='warn'))  # 宏平均（Macro-averaging）是指所有类别的每一个统计指标值的算数平均值
    print("测试集的recall为：")
    print(recall_score(y_test, y_pred, average='macro', zero_division='warn'))
    print("测试集的f1_score为：")
    print(f1_score(y_test, y_pred, average='macro', zero_division='warn'))

'''
path = r"D:\TestFile\WeChat\20200916\MergeWithFeatureAndLabel\merge_text_voice_voicecall.csv"
data = read_data_from_csv(path)
x = data[:, :12]
y = data[:, 12]
print("RF:")
clf_rf_split_train(x, y)
'''

path1 = r"C:\\Users\Wen Ping\Desktop\20200916\Train\MergeWithFeatureAndLabel\merge_text_voice_voicecall.csv"
path2 = r"C:\\Users\Wen Ping\Desktop\20200916\Test\MergeWithFeatureAndLabel\merge_text_voice_voicecall.csv"
data1 = read_data_from_csv(path1)
x_train = data1[:, :12]
y_train = data1[:, 12]
y_train = y_train.astype(np.uint8)
data2 = read_data_from_csv(path2)
x_test = data2[:, :12]
y_test = data2[:, 12]
y_test = y_test.astype(np.uint8)
#xx_train, xx_test, yy_train, yy_test = train_test_split(x_test, y_test, test_size=0.9, random_state=0, stratify=y_test)
print("RF:")
clf_rf_test(x_train, x_test, y_train, y_test)



'''
# 加载svm.pickle
with open(r'D:\TestFile\WeChat\20200916\model\svm_text_voice_voicecall.pickle', 'rb') as fr:
    new_svm = pickle.load(fr)
    y_hat = new_svm.predict(x_test)
    cm = confusion_matrix(y_test, y_hat)  # 生成混淆矩阵 混淆矩阵的第一个值cm[0][0]表示实际为0类，预测为0类的样本数
    # cm2[1][0]表示实际为1类，预测为0类的样本数
    # print(classification_report(y_test, y_hat, target_names=["browsing", "text", "voice", "video"],digits=2))
    # 假设标签1代表browsing、标签2代表text、标签3代表voice、标签4代表video digits可以设置小数点后保留的位数 默认是2
    # 准确率 正确分类的样本数 比上 总样本数
    print("测试集的accuracy为：")
    print(accuracy_score(y_test, y_hat))  # 混淆矩阵对角线元素之和 比上 混淆矩阵所有元素和
    print("测试集的precision为：")
    print(precision_score(y_test, y_hat, average='macro', zero_division='warn'))  # 宏平均（Macro-averaging）是指所有类别的每一个统计指标值的算数平均值
    print("测试集的recall为：")
    print(recall_score(y_test, y_hat, average='macro', zero_division='warn'))
    print("测试集的f1_score为：")
    print(f1_score(y_test, y_hat, average='macro', zero_division='warn'))
'''





