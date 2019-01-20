import numpy as np
import time
from collections import Counter
from knn.knn_base import KNN
from utils.data_generater import random_points
from utils.plot import plot_knn_predict

# kd-tree每个结点中主要包含的数据结构如下
class Node:
    def __init__(self, data, label, depth=0, lchild=None, rchild=None):
        self.data = data
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild
        self.label = label


class KdTree:
    def __init__(self, dataSet, label):
        self.KdTree = None
        self.n = 0
        self.nearest = None
        self.create(dataSet, label)

    # 建立kdtree
    def create(self, dataSet, label, depth=0):
        if len(dataSet) > 0:
            m, n = np.shape(dataSet)
            self.n = n
            axis = depth % self.n
            mid = int(m / 2)
            dataSetcopy = sorted(dataSet, key=lambda x: x[axis])
            node = Node(dataSetcopy[mid], label[mid], depth)
            if depth == 0:
                self.KdTree = node
            node.lchild = self.create(dataSetcopy[:mid], label, depth+1)
            node.rchild = self.create(dataSetcopy[mid+1:], label, depth+1)
            return node
        return None

    # 前序遍历
    def preOrder(self, node):
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    # 搜索kdtree的前count个近的点
    def search(self, x, count = 1):
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        # 初始化n个点，nearest是按照距离递减的方式
        self.nearest = np.array(nearest)

        def recurve(node):
            if node is not None:
                # 计算当前点的维度axis
                axis = node.depth % self.n
                # 计算测试点和当前点在axis维度上的差
                daxis = x[axis] - node.data[axis]
                # 如果小于进左子树，大于进右子树
                if daxis < 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)
                # 计算预测点x到当前点的距离dist
                dist = np.sqrt(np.sum(np.square(x - node.data)))
                for i, d in enumerate(self.nearest):
                    # 如果有比现在最近的n个点更近的点，更新最近的点
                    if d[0] < 0 or dist < d[0]:
                        # 插入第i个位置的点
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis=0)
                        # 删除最后一个多出来的点
                        self.nearest = self.nearest[:-1]
                        break

                # 统计距离为-1的个数n
                n = list(self.nearest[:, 0]).count(-1)
                '''
                self.nearest[-n-1, 0]是当前nearest中已经有的最近点中，距离最大的点。
                self.nearest[-n-1, 0] > abs(daxis)代表以x为圆心，self.nearest[-n-1, 0]为半径的圆与axis
                相交，说明在左右子树里面有比self.nearest[-n-1, 0]更近的点
                '''
                if self.nearest[-n-1, 0] > abs(daxis):
                    if daxis < 0:
                        recurve(node.rchild)
                    else:
                        recurve(node.lchild)

        recurve(self.KdTree)

        # nodeList是最近n个点的
        nodeList = self.nearest[:, 1]

        # knn是n个点的标签
        knn = [node.label for node in nodeList]
        return self.nearest[:, 1], Counter(knn).most_common()[0][0]


class KNNKdTree(KNN):
    def __init__(self, n_neighbors=3, p=2):
        super(KNNKdTree, self).__init__(n_neighbors=n_neighbors, p=p)


    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.kdTree = KdTree(self.X_train, self.y_train)


    def predict(self, point):
        nearest, label = self.kdTree.search(point, self.n)
        # print("nearest", [node.data for node in nearest])
        return nearest, label



    def score(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            _, label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)



def simpleTest():
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    label = [0, 0, 0, 1, 1, 1]
    kdtree = KNNKdTree()
    kdtree.fit(data, label)
    _, predict_label = kdtree.predict([3, 4.5])
    print("predict label:", predict_label)
    # plot_knn_predict(kdtree, data, [3,4.5])

def largeTest():
    N = 400000
    startTime = time.time()
    data = random_points(2, N)
    label = [0] * (N // 2) + [1] * (N // 2)
    kdtree2 = KNNKdTree()
    kdtree2.fit(data, label)
    _, predict_label = kdtree2.predict([0.1, 0.5])  # 四十万个样本点中寻找离目标最近的点

    print("time: %s" % round(time.time() - startTime, 5))
    print("predict label:", predict_label)


def main():
    simpleTest()
    largeTest()

if __name__ == "__main__":
    main()