from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# print(digits.images[0])
images_and_labels = list(zip(digits.images,digits.target))
plt.figure(figsize=(8, 6), dpi=200)
'''enumerate函数
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
'''
for index,(image,label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1) # 2行4列，subplot的index从1开始
    # plt.axis('off')
    plt.imshow(image,cmap='gray_r',interpolation='nearest') # 绘制灰度图，gray_r中的r表示reverse
    plt.title('digit:%i' % label,fontsize=20)
plt.show()
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt #doctest: +SKIP
plt.gray() #doctest: +SKIP
plt.matshow(digits.images[0]) #doctest: +SKIP
plt.show() #doctest: +SKIP

from sklearn.model_selection import train_test_split

# Split arrays or matrices into random train and test subsets
# xtrain是特征向量，ytrain是相应的类别
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.2,
                                                random_state=2)  # random_state 整数代表随机种子

# 使用支持向量机训练模型
from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(Xtrain, Ytrain)

# 模型测试
# 评估模型的准确度
from sklearn.metrics import accuracy_score
#Perform classification on samples in X.
Ypred = clf.predict(Xtest)
print(Ypred)
print(accuracy_score(Ytest, Ypred))

print(clf.score(Xtest, Ytest))

fig, axes = plt.subplots(4, 4, figsize=(8, 8))# 4行4列,每个大小8*8
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32, transform=ax.transAxes,
            color='green' if Ypred[i] == Ytest[i] else 'red')
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# 保存模型参数
from sklearn.externals import joblib
joblib.dump(clf,"dig_svm.pkl")

# 导入模型参数，直接进行预测
clf=joblib.load("dig_svm.pkl")
Ypred=clf.predict(Xtest)
print(clf.score(Xtest, Ytest))

