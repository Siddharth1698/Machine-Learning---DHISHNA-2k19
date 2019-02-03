from sklearn import datasets
import matplotlib.pyplot as plt
iris_flower = datasets.load_iris()
shapr_iris = iris_flower.data.shape
names = iris_flower.feature_names
dataset = iris_flower.data
X = iris_flower.data[:,2:]
y = iris_flower.target

plt.scatter(X[:,0],X[:,1],c = y,cmap=plt.cm.coolwarm)


plt.show()
