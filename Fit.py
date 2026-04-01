import sys
import numpy as np
from loadDataset import loadData
from regTrees import createTree, predict, regErr, regLeaf
from visualize import (
    plot_tree,
    plot_tree_regression,
    show_iris_image,
    plot_regression_results,
)

def fit_regression_tree(dataset_name, ops):
    dataset = loadData(dataset_name)
    data = dataset['data']
    X = data[:, :-1]
    y = data[:, -1]

    tree = createTree(data, leafType=regLeaf, errType=regErr, ops=ops)
    y_pred = predict(tree, X)
    mse = np.mean((y - y_pred) ** 2)
    return dataset, tree, X, y, y_pred, mse

def run_iris():
    sample = np.array([[4.25, 7.9, 5.8, 0.8]], dtype=float) 

    dataset = loadData('iris')
    data = dataset['data']
    #tree = createTree(data, ops=(0, 1))    #Tree without preprunning, overfitted
    tree = createTree(data, ops=(1, 5))     #Tree with preprunning, not overfitted
    print(tree)

    X = data[:, :-1]
    y = data[:, -1]
    accuracy = np.mean(predict(tree, X) == y)
    print(f"Accuracy: {accuracy:.4f}")

    prediction = predict(tree, sample)[0]
    predicted_class = str(dataset['target_names'][int(prediction)])
    print('Prediction for sample ', sample, ': ', predicted_class)
    show_iris_image(predicted_class, sample=sample, feature_names=dataset['feature_names'])

    plot_tree(tree, title='Drzewo decyzyjne - Iris')


def run_sine():
    # _, tree, X, y, y_pred, mse = fit_regression_tree('sinus', ops=(0.01, 1)) #overfitted
    dataset, tree, X, y, y_pred, mse = fit_regression_tree('sinus', ops=(0.01, 10))  #not overfitted
    print(f"MSE (sinus): {mse:.5f}")

    x_vals = X.ravel()
    plot_regression_results(
        x_vals,
        y,
        y_pred,
        title='Regresja CART - aproksymacja sinusa',
        x_label='x',
        data_label='Dane (sin + szum)',
        baseline_x=x_vals,
        baseline_y=np.sin(x_vals),
        baseline_label='Sinus',
    )
    plot_tree_regression(tree, dataset['feature_names'], title='Drzewo regresyjne - Sinus')


def run_diabetes():
    dataset, tree, X, y, y_pred, mse = fit_regression_tree('diabetes', ops=(25, 5)) #Dużo feater'ów, mało danych, ciężko o kompromis
    print(f"MSE (diabetes): {mse:.2f}")
    
    sample = X[0:1] 
    sample_prediction = predict(tree, sample)[0]
    print(f"Prediction for sample: {sample_prediction:.2f} (actual: {y[0]:.2f})")

    plot_tree_regression(tree, dataset['feature_names'], title='Drzewo regresyjne - Diabetes')


def run_friedman_function():
    dataset, tree, X, y, y_pred, mse = fit_regression_tree('friedman_function', ops=(20, 5))
    print(f"MSE (friedman_function): {mse:.2f}")

    sample_idx = np.arange(len(y))
    plot_regression_results(
        sample_idx,
        y,
        y_pred,
        title='Regresja CART - aproksymacja Friedman #1',
        x_label='indeks probki',
    )
    plot_tree_regression(tree, dataset['feature_names'], title='Drzewo regresyjne - Friedman Function')

