import sys
import numpy as np
import matplotlib.pyplot as plt

from loadDataset import loadData
from regTrees import createTree, predict, regErr, regLeaf
from visualize import plot_tree, plot_tree_regression

mode = 'iris'  # iris, sinus, diabetes, funny-function



def run_iris():
    dataset = loadData('iris')
    data = dataset['data']
    #tree = createTree(data, ops=(0, 1))    #Tree without preprunning, overfitted
    tree = createTree(data, ops=(1, 5))     #Tree with preprunning, not overfitted
    print(tree)

    X = data[:, :-1]
    y = data[:, -1]
    accuracy = np.mean(predict(tree, X) == y)
    print(f"Accuracy: {accuracy:.4f}")

    sample = np.array([[4.25, 7.9, 5.8, 0.8]], dtype=float)
    prediction = predict(tree, sample)[0]
    print('Prediction for sample ', sample, ': ', dataset['target_names'][int(prediction)])

    plot_tree(tree, title='Drzewo decyzyjne - Iris')


def run_sinus():
    dataset = loadData('sinus')
    data = dataset['data']

    tree = createTree(data, leafType=regLeaf, errType=regErr, ops=(0.01, 5))
    X = data[:, :-1]
    y = data[:, -1]
    y_pred = predict(tree, X)

    mse = np.mean((y - y_pred) ** 2)
    print(f"MSE (sinus): {mse:.5f}")

    x_vals = X.ravel()
    plt.figure(figsize=(10, 4))
    plt.scatter(x_vals, y, s=10, alpha=0.5, label='Dane (sin + szum)')
    plt.plot(x_vals, y_pred, color='red', linewidth=2, label='Regresja CART')
    plt.plot(x_vals, np.sin(x_vals), color='green', linestyle='--', linewidth=1.5, label='Sinus')
    plt.legend()
    plt.title('Regresja CART - aproksymacja sinusa')
    plt.tight_layout()
    plt.show()


def run_diabetes():
    dataset = loadData('diabetes')
    data = dataset['data']

    tree = createTree(data, leafType=regLeaf, errType=regErr, ops=(5000, 15))
    X = data[:, :-1]
    y = data[:, -1]
    y_pred = predict(tree, X)

    mse = np.mean((y - y_pred) ** 2)
    print(f"MSE (diabetes): {mse:.2f}")
    plot_tree_regression(tree, dataset['feature_names'], title='Drzewo regresyjne - Diabetes')


def run_funny_function():
    print()


if mode == 'iris':
    run_iris()
elif mode == 'sinus':
    run_sinus()
elif mode == 'diabetes':
    run_diabetes()
elif mode == 'funny-function':
    run_funny_function()
else:
    print(f"Błąd, Nieprawidłowy tryb: {mode}")
