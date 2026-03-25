import numpy as np
from loadDataset import loadData
from regTrees import createTree, predict
from visualize import  plot_tree

dataset = loadData('iris')
data=dataset['data']
#tree = createTree(data, ops=(0, 1))    #Tree without preprunning, overfitted
tree = createTree(data, ops=(1, 5))     #Tree with preprunning, not overfitted
print(tree)

X = data[:, :-1]
y = data[:, -1]
accuracy = np.mean(predict(tree, X) == y)
print(f"Accuracy: {accuracy:.4f}")
sample = np.array([[4.25, 7.9, 5.8, 0.8]], dtype=float)
prediction = predict(tree, sample)[0]
print('Prediction for sample ',sample,': ',dataset['target_names'][int(prediction)])
plot_tree(tree)