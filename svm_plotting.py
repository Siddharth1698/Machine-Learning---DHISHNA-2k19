from sklearn import svm
C=1.0
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
iris_flower = datasets.load_iris()
lin_svc = svm.SVC (kernel='linear',C=C)
rbf_syn = svm.SVC (kernel='rbf',gamma=0.7,C=C)
poly_svc = svm.SVC (kernel='poly',degree=3,C=C)

X_train,X_test,y_train,y_test = train_test_split(iris_flower.data,iris_flower.target,test_size = 0.3)

lin_svc.fit(X_train,y_train)
rbf_syn.fit(X_train,y_train)
poly_svc.fit(X_train,y_train)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

lin_predict_result = lin_svc.predict(X_test)
# print(y_test)
# print(lin_predict_result)

rbf_predict_result = rbf_syn.predict(X_test)
# print(y_test)
# print(rbf_predict_result)

poly_predict_result = poly_svc.predict(X_test)
# print(y_test)
# print(poly_predict_result)



# print(accuracy_score(y_test,lin_predict_result,normalize = True))
# print(accuracy_score(y_test,rbf_predict_result,normalize = True))
# print(accuracy_score(y_test,poly_predict_result,normalize = True))

h= 0.02
X = iris_flower.data[:,:2]
Y = iris_flower.target
X_MIN,X_MAX = X[:,0].min()-1,X[:,0].max()+1
Y_MIN,Y_MAX = X[:,0].min()-1,X[:,0].max()+1
XX,YY = np.meshgrid(np.arange(X_MIN,X_MAX,h),np.arange(Y_MIN,Y_MAX,h))

titles = ['SVC with Linear kernel','SVC with RBF kernel','SVC with poly kernel']

for i, clf in enumerate((l, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])


 
    

