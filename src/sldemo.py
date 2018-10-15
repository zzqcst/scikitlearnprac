from sklearn import datasets
import matplotlib.pyplot as plt
digits=datasets.load_digits()
# print(digits.images[0])
# images_and_labels = list(zip(digits.images,digits.target))
# plt.figure(figsize=(8,6),dpi=200)
# '''enumerate函数
# >>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# >>> list(enumerate(seasons))
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
# '''
# for index,(image,label) in enumerate(images_and_labels[:8]):
#     plt.subplot(2,4,index+1) # 2行4列，subplot的index从1开始
#     # plt.axis('off')
#     plt.imshow(image,cmap='gray_r',interpolation='nearest') # 绘制灰度图，gray_r中的r表示reverse
#     plt.title('digit:%i' % label,fontsize=20)
# plt.show()
# from sklearn.datasets import load_digits
# digits = load_digits()
# print(digits.data.shape)
# import matplotlib.pyplot as plt #doctest: +SKIP
# plt.gray() #doctest: +SKIP
# plt.matshow(digits.images[0]) #doctest: +SKIP
# plt.show() #doctest: +SKIP

from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets
Xtrain,Xtest,Ytrain,Ytest = train_test_split(digits.data,digits.target,test_size=0.2,random_state=2)# random_state 整数代表随机种子

# 使用支持向量机训练模型
from sklearn import svm
clf = svm.SVC(gamma=0.001,C=100.)
clf.fit(Xtrain,Ytrain)

# 模型测试
print(clf.score(Xtest, Ytest))
