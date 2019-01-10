from collections import defaultdict
from random import uniform
from sklearn import datasets
from utils.data_generater import *
import time



# 输入：points是一类的点，维度相同
# 输出：这些点的中心点
def point_avg(points):
    return np.mean(points, axis=0)


#输入：data_set是数据集的点，assignments是每个点在当前归为的类别
#输出：新的中心点list
def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        newCenter = point_avg(points)
        centers.append(newCenter)

    return centers

#输入：data_set是数据集的点，assignments是每个点在当前归为的类别
#输出：新的误差值
def update_error(data_set, assignments):
    new_means = defaultdict(list)
    error = 0
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        newCenter = point_avg(points)
        error += np.sqrt(np.sum(np.square(points - newCenter)))

    return error

#输入：data_set原始数据集，centers所有的中心点
#输出：每个点对应的聚类类别
def assign_points(data_set, centers):
    assignments = []
    for point in data_set:
        shortest = float("inf")  # 正无穷
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


#k个数据点，随机生成
def generate_k(data_set, k):
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        error = update_error(dataset, assignments)
        print("error", error)
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return zip(assignments, dataset)


iris = datasets.load_iris()
for k in k_means(iris.data, 3):
    print(k)


pointList = []
numPoints = 10000
dim = 1000
numClusters = 10
k = 0
for i in range(0,numClusters):
    num = int(numPoints/numClusters)
    p = makeRandomPoint(num,dim,k)
    k += 5
    pointList += p.tolist()

start = time.time()
config= k_means(np.array(pointList), numClusters)
print("Time taken:",time.time() - start)