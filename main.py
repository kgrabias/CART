import numpy as np
from loadDataset import loadData
from regTrees import createTree

data = loadData('iris')['data']
#tree = createTree(data, ops=(0, 1))    #Tree without preprunning, overfitted
tree = createTree(data, ops=(1, 4))     #Tree with preprunning, not overfitted
print(tree)