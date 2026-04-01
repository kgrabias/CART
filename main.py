import numpy as np
from loadDataset import loadData
from regTrees import createTree, predict, regErr, regLeaf
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
dataset_sin = loadData('sinus')
data_sin = dataset_sin['data']

tree_sin = createTree(data_sin, leafType=regLeaf, errType=regErr, ops=(0.01, 5))

X_sin = data_sin[:, :-1]
y_sin = data_sin[:, -1]
y_pred = predict(tree_sin, X_sin)

mse = np.mean((y_sin - y_pred) ** 2)
print(f"\nMSE (sinus): {mse:.5f}")

plt.figure(figsize=(10, 4))
plt.scatter(X_sin, y_sin, s=10, alpha=0.5, label="Dane (sin + szum)")
plt.plot(X_sin, y_pred, color="red", linewidth=2, label="Regresja CART")
plt.plot(X_sin, np.sin(X_sin), color="green", linestyle="--", linewidth=1.5, label="Sinus")
plt.legend()
plt.title("Regresja CART – aproksymacja sinusa")
plt.tight_layout()
plt.show()