from AddLabel import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import *


path = r"D:\TestFile\merge.csv"
data = read_data_from_csv(path)
x = data[:, :10]
y = data[:, 10]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

clf = SVC()
clf.fit(x_train, y_train)




y_predict = clf.predict(x_test)
ov_acc = accuracy_score(y_test, y_predict)
print("accuracy")
print(ov_acc)

av_pre = precision_score(y_test, y_predict, average=None)
print("each_class_precision")
print(av_pre)
print("average_precision")
print(np.mean(av_pre))

