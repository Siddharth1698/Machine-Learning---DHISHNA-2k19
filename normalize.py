import numpy as np 
from sklearn import preprocessing


input_data = np.array([[1,2],[3,4],[5,6]])
data_normalized_l1 = preprocessing.normalize(input_data,'l1')
print(data_normalized_l1) 
data_normalized_l2 = preprocessing.normalize(input_data,'l2')
print(data_normalized_l2) 