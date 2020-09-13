from AddLabel import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier




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

def ClfRF(x, y):

    acc = []  # 初始化acc
    f1_sco = []
    pre = []
    rec = []
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    skfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skfold.split(x, y):  # 对数据建立5折交叉验证的划分
        # for test_index,train_index in New_sam.split(Sam):  #默认第一个参数是训练集，第二个参数是测试集
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print('训练集数量:', x_train.shape, '测试集数量:', x_test.shape)  # 结果表明每次划分的数量
        clf = RandomForestClassifier(n_estimators=100, max_features='log2')  # 定义随机森林分类器， 树的个数为50
        clf.fit(x_train, y_train)  # 训练分类器
        y_hat = clf.predict(x_test)  # 预测测试集的类别
        cm = confusion_matrix(y_test, y_hat)  # 生成混淆矩阵 混淆矩阵的第一个值cm[0][0]表示实际为0类，预测为0类的样本数
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


def ClfKNN(x, y):

    acc = []  # 初始化acc
    f1_sco = []
    pre = []
    rec = []
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    skfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skfold.split(x, y):  # 对数据建立5折交叉验证的划分
        # for test_index,train_index in New_sam.split(Sam):  #默认第一个参数是训练集，第二个参数是测试集
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print('训练集数量:', x_train.shape, '测试集数量:', x_test.shape)  # 结果表明每次划分的数量
        clf = KNeighborsClassifier(n_neighbors=1)  # 定义KNN分类器， 近邻的个数为3
        clf.fit(x_train, y_train)  # 训练分类器
        y_hat = clf.predict(x_test)  # 预测测试集的类别
        cm = confusion_matrix(y_test, y_hat)  # 生成混淆矩阵 混淆矩阵的第一个值cm[0][0]表示实际为0类，预测为0类的样本数
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

path = r"D:\TestFile\merge.csv"
data = read_data_from_csv(path)
x = data[:, :12]
y = data[:, 12]

print("RF:")
ClfRF(x, y)

# print("KNN:")
# ClfKNN(x, y)





