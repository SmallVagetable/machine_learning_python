
import random

from collections import defaultdict

from sklearn.cluster import KMeans
import numpy as np

from utils.misc_utils import distance, check_random_state
from utils.data_generater import makeRandomPoint



class KMeansBase(object):

    def __init__(self, n_clusters = 8, init="random", max_iter = 300, random_state = None):
        self.k = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state


    def fit(self, dataset):
        self.dataset = np.array(dataset)
        k_points = self._init_centroids(dataset)
        assignments = self.assign_points(dataset, k_points)
        old_assignments = None
        for i in range(self.max_iter):
            if assignments == old_assignments:
                break
            error = self.update_error(dataset, assignments)
            print("error", error)
            new_centers = self.update_centers(dataset, assignments)
            old_assignments = assignments
            assignments = self.assign_points(dataset, new_centers)
        return zip(assignments, dataset)


    # k个数据点，随机生成
    def _init_centroids(self, dataset):
        random_state = check_random_state(self.random_state)
        n_samples = dataset.shape[0]
        if self.init == "random":
            seeds = random_state.permutation(n_samples)[:self.k]
            centers = dataset[seeds]
        elif self.init == "k-means++":
            centers = []

        return centers


    # 输入：points是一个聚类的点，维度相同
    # 输出：这些点的中心点
    def point_avg(self, points):
        return np.mean(points, axis=0)


    #输入：data_set是数据集的点，assignments是每个点在当前归为的类别
    #输出：新的中心点list
    def update_centers(self, dataset, assignments):
        new_means = defaultdict(list)
        centers = []
        for assignment, point in zip(assignments, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = self.point_avg(points)
            centers.append(newCenter)

        return centers

    #输入：data_set是数据集的点，assignments是每个点在当前归为的类别
    #输出：新的误差值
    def update_error(self, dataset, assignments):
        new_means = defaultdict(list)
        error = 0
        for assignment, point in zip(assignments, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            newCenter = self.point_avg(points)
            error += np.sqrt(np.sum(np.square(points - newCenter)))

        return error

    #输入：data_set原始数据集，centers所有的中心点
    #输出：每个点对应的聚类类别
    def assign_points(self, dataset, centers):
        assignments = []
        for point in dataset:
            shortest = float("inf")  # 正无穷
            shortest_index = 0
            for i in range(len(centers)):
                val = distance(point, centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments






if __name__ == "__main__":

    iris = datasets.load_iris()
    km = KMeansBase(3)
    for k in km.fit(iris.data):
        print(k)

    kmeans = KMeans(init='k-means++', n_clusters= 10, n_init=10)

# pointList = []
# numPoints = 10000
# dim = 1000
# numClusters = 10
# k = 0
# for i in range(0,numClusters):
#     num = int(numPoints/numClusters)
#     p = makeRandomPoint(num,dim,k)
#     k += 5
#     pointList += p.tolist()
#
# start = time.time()
# config= k_means(np.array(pointList), numClusters)
# print("Time taken:",time.time() - start)