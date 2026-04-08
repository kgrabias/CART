import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree as sklearn_plot_tree
from loadDataset import loadData
from regTrees import classErr, classLeaf, createTree, predict, regErr, regLeaf
from visualize import (
    plot_tree,
    plot_tree_regression,
    show_iris_image,
    plot_regression_results,
)

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def timed_custom_tree(data, ops, leaf_type, err_type):
    start = time.perf_counter()
    tree = createTree(data, leafType=leaf_type, errType=err_type, ops=ops)
    return tree, time.perf_counter() - start


def timed_sklearn_fit(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def print_comparison_metrics(mode_name, mse_custom, mse_sklearn, custom_build_time, sklearn_build_time):
    print(f"MSE ({mode_name}, CART własny): {mse_custom:.5f}")
    print(f"MSE ({mode_name}, CART scikit-learn): {mse_sklearn:.5f}")
    print(f"Czas budowy drzewa ({mode_name}, CART własny): {custom_build_time * 1000:.2f} ms")
    print(f"Czas budowy drzewa ({mode_name}, CART scikit-learn): {sklearn_build_time * 1000:.2f} ms")


def friedman_true_function(X):
    X = np.asarray(X)
    # Friedman #1 depends on the first 5 features.
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def plot_sklearn_tree(model, feature_names, title, class_names=None):
    plt.figure(figsize=(13, 7))
    sklearn_plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        fontsize=8,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def run_regression_mode(
    dataset_name,
    display_name,
    custom_ops,
    sklearn_params,
    x_values,
    x_label,
    data_label,
    baseline_x=None,
    baseline_y=None,
    baseline_label='Funkcja odniesienia',
    show_regression_plots=True,
):
    dataset = loadData(dataset_name)
    data = dataset['data']
    X = data[:, :-1]
    y = data[:, -1]

    custom_tree, custom_build_time = timed_custom_tree(
        data,
        ops=custom_ops,
        leaf_type=regLeaf,
        err_type=regErr,
    )
    y_pred_custom = predict(custom_tree, X)
    mse_custom = mse(y, y_pred_custom)

    sklearn_tree = DecisionTreeRegressor(random_state=42, **sklearn_params)
    sklearn_tree, sklearn_build_time = timed_sklearn_fit(sklearn_tree, X, y)
    y_pred_sklearn = sklearn_tree.predict(X)
    mse_sklearn = mse(y, y_pred_sklearn)

    print_comparison_metrics(
        display_name,
        mse_custom,
        mse_sklearn,
        custom_build_time,
        sklearn_build_time,
    )

    if show_regression_plots:
        plot_regression_results(
            x_values,
            y,
            y_pred_custom,
            title=f'Regresja CART (własna implementacja) - {display_name}',
            x_label=x_label,
            data_label=data_label,
            baseline_x=baseline_x,
            baseline_y=baseline_y,
            baseline_label=baseline_label,
            model_label='CART (własny)',
        )
        plot_regression_results(
            x_values,
            y,
            y_pred_sklearn,
            title=f'Regresja CART (scikit-learn) - {display_name}',
            x_label=x_label,
            data_label=data_label,
            baseline_x=baseline_x,
            baseline_y=baseline_y,
            baseline_label=baseline_label,
            model_label='CART (scikit-learn)',
        )

    plot_tree_regression(custom_tree, dataset['feature_names'], title=f'Drzewo regresyjne - {display_name}')
    plot_sklearn_tree(
        sklearn_tree,
        feature_names=dataset['feature_names'],
        title=f'Drzewo regresyjne - {display_name} (scikit-learn)',
    )

    return X, y, custom_tree, sklearn_tree

def run_iris():
    sample = np.array([[4.25, 7.9, 5.8, 0.8]], dtype=float) 

    dataset = loadData('iris')
    data = dataset['data']
    tree, custom_build_time = timed_custom_tree(data, ops=(1, 5), leaf_type=classLeaf, err_type=classErr)

    X = data[:, :-1]
    y = data[:, -1]
    y_pred_custom = predict(tree, X)
    mse_custom = mse(y, y_pred_custom)
    accuracy = np.mean(y_pred_custom == y)

    sklearn_tree = DecisionTreeClassifier(
        criterion='gini',
        min_samples_split=5,
        random_state=42,
    )
    sklearn_tree, sklearn_build_time = timed_sklearn_fit(sklearn_tree, X, y)
    y_pred_sklearn = sklearn_tree.predict(X)
    mse_sklearn = mse(y, y_pred_sklearn)
    sklearn_accuracy = np.mean(y_pred_sklearn == y)

    print_comparison_metrics('iris', mse_custom, mse_sklearn, custom_build_time, sklearn_build_time)
    print(f"Accuracy (iris, CART własny): {accuracy:.4f}")
    print(f"Accuracy (iris, CART scikit-learn): {sklearn_accuracy:.4f}")

    prediction = predict(tree, sample)[0]
    predicted_class = str(dataset['target_names'][int(prediction)])
    prediction_sklearn = sklearn_tree.predict(sample)[0]
    predicted_class_sklearn = str(dataset['target_names'][int(prediction_sklearn)])
    print('Prediction for sample (CART własny) ', sample, ': ', predicted_class)
    print('Prediction for sample (CART scikit-learn) ', sample, ': ', predicted_class_sklearn)
    show_iris_image(predicted_class, sample=sample, feature_names=dataset['feature_names'])

    plot_tree(tree, title='Drzewo decyzyjne - Iris')
    plot_sklearn_tree(
        sklearn_tree,
        feature_names=dataset['feature_names'],
        class_names=dataset['target_names'],
        title='Drzewo decyzyjne - Iris (scikit-learn)',
    )


def run_sine():
    dataset = loadData('sinus')
    X = dataset['data'][:, :-1]
    x_vals = X.ravel()
    run_regression_mode(
        dataset_name='sinus',
        display_name='sinus',
        custom_ops=(0.01, 10),
        sklearn_params={'min_impurity_decrease': 0.01, 'min_samples_split': 10},
        x_values=x_vals,
        x_label='x',
        data_label='Dane (sin + szum)',
        baseline_x=x_vals,
        baseline_y=np.sin(x_vals),
        baseline_label='Sinus',
        show_regression_plots=True,
    )


def run_diabetes():
    X, y, tree, sklearn_tree = run_regression_mode(
        dataset_name='diabetes',
        display_name='diabetes',
        custom_ops=(25, 5),
        sklearn_params={'min_impurity_decrease': 25, 'min_samples_split': 5},
        x_values=np.arange(loadData('diabetes')['data'].shape[0]),
        x_label='indeks probki',
        data_label='Dane',
        show_regression_plots=False,
    )
    
    sample = X[0:1] 
    sample_prediction = predict(tree, sample)[0]
    sample_prediction_sklearn = sklearn_tree.predict(sample)[0]
    print(f"Prediction for sample (CART własny): {sample_prediction:.2f} (actual: {y[0]:.2f})")
    print(f"Prediction for sample (CART scikit-learn): {sample_prediction_sklearn:.2f} (actual: {y[0]:.2f})")


def run_friedman_function():
    dataset = loadData('friedman_function')
    X = dataset['data'][:, :-1]
    sample_idx = np.arange(len(X))
    true_y = friedman_true_function(X)

    run_regression_mode(
        dataset_name='friedman_function',
        display_name='friedman_function',
        custom_ops=(1, 5),
        sklearn_params={'min_impurity_decrease': 0.1, 'min_samples_split': 5},
        x_values=sample_idx,
        x_label='indeks probki',
        data_label='Dane (Friedman + szum)',
        baseline_x=sample_idx,
        baseline_y=true_y,
        baseline_label='Funkcja Friedmana (bez szumu)',
        show_regression_plots=True,
    )

