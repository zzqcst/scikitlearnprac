from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
# 生成数据
centers = [[-2,2],[2,2],[0,4]]
'''
生成60个训练样本，分布在以centers参数指定中心点周围，cluster_std是标准差，生成的训练数据集X，类别标记y
'''
X,y=make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.60)
print(X)
print(y)



from sklearn.neighbors import KNeighborsClassifier
# 模型训练
k=5
clf=KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)

# 进行预测
X_sample=[[2,2]]
y_sample=clf.predict(X_sample)
print(y_sample[0])
'''
print(neighbors)
[[16 20 48  6 23]]
'''
neighbors = clf.kneighbors(X_sample,return_distance=False) # 取出来的点是训练样本X里的索引，从0开始

plt.figure(figsize=(16,10),dpi=144)
c=np.array(centers)
'''
s:marker size
c:color
cmap:颜色映射表
'''
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap='cool')
plt.scatter(c[:,0],c[:,1],s=100,marker="^",c="orange")
plt.scatter(X_sample[0][0],X_sample[0][1],marker='x',c=np.asarray(y_sample[0]),cmap='cool',s=100)
for i in neighbors[0]: #遍历索引
    plt.plot([X[i][0],X_sample[0][0]],[X[i][1],X_sample[0][1]],'k--',linewidth=0.6) # 预测点与距离最近的5个样本的连线
plt.show()