
import numpy as np


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


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def chooseBestSplit(dataSet, ops=(1, 4)):
    tolS, tolN = ops

    if len(set(dataSet[:, -1])) == 1:
        return None, classLeaf(dataSet)

    m, n = dataSet.shape
    S = classErr(dataSet)

    bestS = np.inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n - 1):
        values = sorted(set(dataSet[:, featIndex]))

        for i in range(len(values) - 1):
            splitVal = (values[i] + values[i+1]) / 2

            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)

            if len(mat0) < tolN or len(mat1) < tolN:
                continue

            newS = classErr(mat0) + classErr(mat1)

            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    if bestIndex is None:
        return None, classLeaf(dataSet)

    if (S - bestS) < tolS:
        return None, classLeaf(dataSet)

    return bestIndex, bestValue