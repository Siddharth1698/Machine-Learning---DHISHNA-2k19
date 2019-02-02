import numpy as np 
from sklearn import preprocessing


input_data = ["apple","orange","grapes"]
label_encode = preprocessing.LabelEncoder()
label_encode.fit(input_data)
encoded_labels = label_encode.transform(input_data)
for i,item in enumerate(label_encode.classes_): 
	print (item, '->', i)

print ("Labels =", input_data)
print ("Encoded labels =", list (encoded_labels))
