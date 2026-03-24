import numpy as np
class Tree:
    def __init__(self, root=None):
        self.root = root

class treeNode:
    def __init__(self, left, right, feature, value):
        self.rightBranch = right
        self.leftBranch = left
        self.featureToSplitOn = feature
        self.valueOfSplit = value

def binSplitDataSet(dataSet, feature, value):
    aboveSet = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    belowEqualSet = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return aboveSet,belowEqualSet


# Find the best feature to split on:
# If we can’t split the data, this node becomes a leaf node
# Make a binary split of the data
# Call createTree() on the right split of the data
# Call createTree() on the left split of the data

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: 
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
