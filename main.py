import numpy as np
from loadDataset import loadData
from regTrees import createTree

data = loadData('iris')
tree = createTree(data, ops=(1, 4))
print(tree)