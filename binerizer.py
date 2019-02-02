import numpy as np 
from sklearn import preprocessing


input_data = np.array([[1,2],[3,4],[5,6]])

data_binerizer = preprocessing.Binarizer(threshold=4)
binerized = data_binerizer.transform(input_data)
print(binerized)