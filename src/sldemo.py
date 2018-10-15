from sklearn import datasets
import matplotlib.pyplot as plt
digits=datasets.load_digits()
images_and_labels = list(zip(digits.images,digits.target))
plt.figure(figsize=(8,6),dpi=200)
for index,(image,label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('digit:%i' % label,fontsize=20)
plt.show()
# from sklearn.datasets import load_digits
# digits = load_digits()
# print(digits.data.shape)
# import matplotlib.pyplot as plt #doctest: +SKIP
# plt.gray() #doctest: +SKIP
# plt.matshow(digits.images[0]) #doctest: +SKIP
# plt.show() #doctest: +SKIP