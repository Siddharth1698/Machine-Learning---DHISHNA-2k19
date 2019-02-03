from sklearn import datasets
iris_flower = datasets.load_iris()
shapr_iris = iris_flower.data.shape
names = iris_flower.feature_names
dataset = iris_flower.data

# print(iris_flower)
# print(shapr_iris)
# print(names)
# print(dataset)
# print (iris_flower.target_names)
# print (iris_flower.target)
# print(iris_flower.target.shape)

print(iris_flower['DESCR'])