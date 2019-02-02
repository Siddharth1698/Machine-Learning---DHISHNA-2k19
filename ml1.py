import numpy as np 
from sklearn import preprocessing

input_data = np.array([[1,2],[3,4],[5,6]])
print(input_data)
print(input_data.mean(axis=0))
print(input_data.std(axis=0))
input_data_scaled = preprocessing.scale(input_data)
print(input_data_scaled)
scaled_mean = (input_data_scaled.mean(axis=0))
print(scaled_mean)