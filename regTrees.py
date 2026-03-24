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


def classLeaf(dataSet):
    labels = dataSet[:, -1]         
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)  


def classErr(dataSet):

    labels = dataSet[:, -1]
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    gini = 1.0 - sum((c / n) ** 2 for c in counts.values())
    return gini * n


def chooseBestSplit(dataSet, leafType=classLeaf, errType=classErr, ops=(1, 4)):
    tolS, tolN = ops

    if len(set(dataSet[:, -1])) == 1:
        return None, leafType(dataSet)

    m, n = dataSet.shape
    S = errType(dataSet)

    bestS = np.inf
    bestIndex = None
    bestValue = None

    for featIndex in range(n - 1):
        values = sorted(set(dataSet[:, featIndex]))

        for i in range(len(values) - 1):
            splitVal = (values[i] + values[i+1]) / 2

            aboveSet,belowEqualSet = binSplitDataSet(dataSet, featIndex, splitVal)

            if len(aboveSet) < tolN or len(belowEqualSet) < tolN:
                continue

            newS = errType(aboveSet) + errType(belowEqualSet)

            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    if bestIndex is None:
        return None, leafType(dataSet)

    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    return bestIndex, bestValue

# Find the best feature to split on:
# If we can’t split the data, this node becomes a leaf node
# Make a binary split of the data
# Call createTree() on the right split of the data
# Call createTree() on the left split of the data
def createTree(dataSet, leafType=classLeaf, errType=classErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
