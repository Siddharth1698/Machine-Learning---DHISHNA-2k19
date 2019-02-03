
h = .02  # step size in the mesh
from sklearn import datasets
from sklearn import svm
C=1.0
import numpy as np
import matplotlib.pyplot as plt

iris_dataset = datasets.load_iris()



x = iris_dataset.data[:, :2]
y = iris_dataset.target

# title for the plots
titles = ['SVC with Linear kernel',
          'SVC with RBF kernel',
          'SVC with Polynomial (degree 3) kernel']

from sklearn.model_selection import train_test_split
iris_flower = datasets.load_iris()
lin_svc = svm.SVC (kernel='linear',C=C)
rbf_syn = svm.SVC (kernel='rbf',gamma=1.3,C=C)
poly_svc = svm.SVC (kernel='poly',degree=3,C=C)

x_train,x_test,y_train,y_test = train_test_split(iris_flower.data,iris_flower.target,test_size = 0.3)

lin_svc.fit(x_train,y_train)
rbf_syn.fit(x_train,y_train)
poly_svc.fit(x_train,y_train)
# create a mesh to plot in
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
for i, clf in enumerate((lin_svc, rbf_syn, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # Plot also the training points
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])


 
    
