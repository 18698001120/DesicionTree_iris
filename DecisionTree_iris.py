from sklearn import datasets
import  numpy as np

'''sklearn中自带了一些数据集，比如iris数据集，iris数据中data存储花瓣长宽和
花萼长宽，target存储花的分类，山鸢尾(setosa)、杂色鸢尾(versicolor)以及维
吉尼亚鸢尾(virginica)分别存储为数字0，1，2。这里使用鸢尾花的全部特征作为分类
标准。'''

iris = datasets.load_iris()
X = iris.data
y = iris.target

'''train_test_split将数据集分为训练集和测试集，test_size参数决定测试集
的比例。random_state参数是随机数生成种子，在分类前将数据打乱，保证数据的可
重复利用。stratify保证训练集和测试集中花的三大类的比例与输入比例相同。
其中X_train,X_test，y_train，y_test分别表示训练集的分类特征，测试集的分
类特征， 训练集的类别标签和测试集的类别标签。'''

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,
        test_size = 0.3,random_state = 4, stratify = y)

'''运用sklearn preprocessing模块的StandardScaler类对特征值进行标准化。
fit 函数计算平均值和标准差，而transform 函数运用 fit 函数计算的均值和标
准差进行数据的标准化。'''

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'训练模型'
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train_std,y_train)
y_pred = model.predict(X_test_std)

'计算模型准确率'
from sklearn.metrics import accuracy_score

miss_classified = (y_pred != y_test).sum() / y_test.sum()
print("MissClassified: ",miss_classified)
print('Accuracy : % .2f' % accuracy_score(y_pred,y_test))