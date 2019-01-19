import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_decision_regions(model, X, y, resolution=0.02):
    """
    拟合效果可视化
    :param X:training sets
    :param y:training labels
    :param resolution:分辨率
    :return:None
    """
    # initialization colors map
    colors = ['red', 'blue']
    markers = ['o', 'x']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision regions
    x1_max, x1_min = max(X[:, 0]) + 1, min(X[:, 0]) - 1
    x2_max, x2_min = max(X[:, 1]) + 1, min(X[:, 1]) - 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.show()


def plot_knn_predict(model, dataset, label, x):
    dataset = np.array(dataset)
    plt.scatter(x[0], x[1], c="r", marker='*', s = 40)  # 测试点
    near, predict_label = model.predict(x)  # 设置临近点的个数
    plt.scatter(dataset[:,0], dataset[:,1], c=label, s = 50)  # 画所有的数据点
    for n in near:
        print(n)
        plt.scatter(n.data[0], n.data[1], c="r", marker='+', s = 40)  # k个最近邻点
    plt.show()