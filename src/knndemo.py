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
plt.figure(figsize=(10,20),dpi=144)
c=np.array(centers)
'''
s:marker size
c:color
cmap:颜色映射表
'''
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap='cool')
plt.show()