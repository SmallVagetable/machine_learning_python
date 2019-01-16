from collections import defaultdict
import time

from sklearn.cluster import KMeans
from sklearn import datasets

import numpy as np

from utils.misc_utils import distance, check_random_state, sortLabel



class KMeansBase(object):

    def __init__(self, n_clusters = 8, init = "random", max_iter = 300, random_state = None, n_init = 10, tol = 1e-4):
        self.k = n_clusters # 聚类个数
        self.init = init # 输出化方式
        self.max_iter = max_iter # 最大迭代次数
        self.random_state = check_random_state(random_state) #随机数
        self.n_init = n_init # 进行多次聚类，选择最好的一次
        self.tol = tol # 停止聚类的阈值

    # fit对train建立模型
    def fit(self, dataset):
        self.tol = self._tolerance(dataset, self.tol)

        bestError = None
        bestCenters = None
        bestLabels = None
        for i in range(self.n_init):
            labels, centers, error = self._kmeans(dataset)
            if bestError == None or error < bestError:
                bestError = error
                bestCenters = centers
                bestLabels = labels
        self.centers = bestCenters
        return bestLabels, bestCenters, bestError

    # predict根据训练好的模型预测新的数据
    def predict(self, X):
        return self.update_labels_error(X, self.centers)[0]

    # 合并fit和predict
    def fit_predict(self, dataset):
        self.fit(dataset)
        return self.predict(dataset)

    # kmeans的主要方法，完成一次聚类的过程
    def _kmeans(self, dataset):
        self.dataset = np.array(dataset)
        bestError = None
        bestCenters = None
        bestLabels = None
        centerShiftTotal = 0
        centers = self._init_centroids(dataset)

        for i in range(self.max_iter):
            oldCenters = centers.copy()
            labels, error = self.update_labels_error(dataset, centers)
            centers = self.update_centers(dataset, labels)

            if bestError == None or error < bestError:
                bestLabels = labels.copy()
                bestCenters = centers.copy()
                bestError = error

            ## oldCenters和centers的偏移量
            centerShiftTotal = np.linalg.norm(oldCenters - centers) ** 2
            if centerShiftTotal <= self.tol:
                break

        #由于上面的循环，最后一步更新了centers，所以如果和旧的centers不一样的话，再更新一次labels，error
        if centerShiftTotal > 0:
            bestLabels, bestError = self.update_labels_error(dataset, bestCenters)

        return bestLabels, bestCenters, bestError


    # k个数据点，随机生成
    def _init_centroids(self, dataset):
        n_samples = dataset.shape[0]
        centers = []
        if self.init == "random":
            seeds = self.random_state.permutation(n_samples)[:self.k]
            centers = dataset[seeds]
        elif self.init == "k-means++":
            pass
        return np.array(centers)


    # 把tol和dataset相关联
    def _tolerance(self, dataset, tol):
        variances = np.var(dataset, axis=0)
        return np.mean(variances) * tol


    # 更新每个点的标签，和计算误差
    def update_labels_error(self, dataset, centers):
        labels = self.assign_points(dataset, centers)
        new_means = defaultdict(list)
        error = 0
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = np.mean(points, axis=0)
            error += np.sqrt(np.sum(np.square(points - newCenter)))

        return labels, error

    # 更新中心点
    def update_centers(self, dataset, labels):
        new_means = defaultdict(list)
        centers = []
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = np.mean(points, axis=0)
            centers.append(newCenter)

        return np.array(centers)


    # 分配每个点到最近的center
    def assign_points(self, dataset, centers):
        labels = []
        for point in dataset:
            shortest = float("inf")  # 正无穷
            shortest_index = 0
            for i in range(len(centers)):
                val = distance(point[np.newaxis], centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            labels.append(shortest_index)
        return labels


if __name__ == "__main__":
    # 用iris数据集测试准确度和速度
    iris = datasets.load_iris()
    km = KMeansBase(3)
    startTime = time.time()
    labels = km.fit_predict(iris.data)
    print("km time", time.time() - startTime)
    print(np.array(sortLabel(labels)))

    kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
    startTime = time.time()
    label = kmeans.fit_predict(iris.data)
    print("sklearn time", time.time() - startTime)
    print(sortLabel(label))