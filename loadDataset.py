import numpy as np
from sklearn.datasets import load_iris, load_diabetes
def loadData(dataset):
	if dataset=='iris':
		data = load_iris(return_X_y=True)
	if dataset=='diabetes':
		data = load_diabetes()
	return data

# For debugging
def printIrisData():
	dataset=loadData('iris')
	data = dataset[0]
	target=dataset[1]
	target_names = ['setosa', 'versicolor', 'virginica']
	for i in range (0,149):
		print('data: ',data[i],'target: ',target_names[int(target[i])])


