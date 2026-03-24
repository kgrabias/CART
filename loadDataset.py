import numpy as np
from sklearn.datasets import load_iris, load_diabetes
def loadData(dataset):
	if dataset=='iris':
		X, y = load_iris(return_X_y=True)
		data = reshapeData(X, y)
	if dataset=='diabetes':
		data = load_diabetes()
	return data


def reshapeData(X, y):
	return np.hstack([X, y.reshape(-1, 1)])


# For debugging
def printIrisData():
	dataset=loadData('iris')
	data = dataset[0]
	target=dataset[1]
	target_names = ['setosa', 'versicolor', 'virginica']
	for i in range (0,149):
		print('data: ',data[i],'target: ',target_names[int(target[i])])



