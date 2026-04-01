import numpy as np
from sklearn.datasets import load_iris, load_diabetes
def loadData(choice):
	if choice == 'sinus':
		np.random.seed(42)
		X = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
		y = np.sin(X).ravel() + np.random.normal(0, 0.1, 200)
		data = reshapeData(X, y)
		return {
        "data": data,
        "feature_names": ["x"],
        "target_names": None
    }
	if choice=='iris':
		dataSet= load_iris()
		target_names = dataSet.target_names.tolist()
	elif choice=='diabetes':
		dataSet = load_diabetes()
	else:
		return None
	X = dataSet.data 	#ndarray
	y = dataSet.target 	#ndarray
	data = reshapeData(X, y)
	feature_names = dataSet.feature_names 	#list
	newDataSet={
			"data" : data,
			"feature_names" : feature_names
			}
	if choice=='iris':
		newDataSet.update({
			"target_names" : target_names
		})
	return newDataSet
	

def reshapeData(X, y):
	return np.hstack([X, y.reshape(-1, 1)])