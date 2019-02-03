X_train = [[5.0,45],[5.11,26],[5.6,30],[5.9,34],[4.8,40],[5.8,36],[5.3,19],[5.8,28],[5.5,23],[5.6,32]]

Y_train = [77,47,55,59,72,60,40,60,45,58]
X_test = [[5.5,38]]



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,Y_train)
pred = knn.predict(X_test)
print(pred)
