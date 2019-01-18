import numpy as np
import time
from collections import Counter
from collections import namedtuple
from knn.knn_base import KNN
from utils.data_generater import random_points

# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right, label):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree
        self.label = label # 该节点的标签
        self.visit = False


    def setVisit(self, isVisite):
        self.visit = isVisite



class KdTree(object):
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        self.root = self.createNode(0, X, y)  # 从第0维分量开始构建kd树,返回根节点


    def createNode(self, split, data_set, label):  # 按第split维划分数据集exset创建KdNode
        if not data_set:  # 数据集为空
            return None
        k = len(data_set[0])  # 数据维度
        # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
        # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
        # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
        data_set.sort(key=lambda x: x[split])
        split_pos = len(data_set) // 2  # //为Python中的整数除法
        median = data_set[split_pos]  # 中位数分割点
        split_next = (split + 1) % k  # cycle coordinates

        # 递归的创建kd树
        return KdNode(median, split,
                      self.createNode(split_next, data_set[:split_pos], label),  # 创建左子树
                      self.createNode(split_next, data_set[split_pos + 1:], label), label[split_pos])  # 创建右子树

    # KDTree的前序遍历
    def preorder(self, root = "root"):
        if root == "root":
            root = self.root
        print(root.dom_elt)
        if root.left:  # 节点不为空
            self.preorder(root.left)
        if root.right:
            self.preorder(root.right)


class KNNKdTree(KNN):
    def __init__(self, n_neighbors=3, p=2):
        super(KNNKdTree, self).__init__(n_neighbors=n_neighbors, p=p)


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.kdTree = KdTree(self.X_train, self.y_train)


    def predict(self, point):
        knn_list = []
        for i in range(self.n):
            node = self.find_nearest(self.kdTree, point)
            node.nearest_node.setVisit(True)
            knn_list.append(node.nearest_node)
            # nearest_node = node.nearest_node
            # print("node point = %s label = %s, dist = %s"%([round(d, 5) for d in nearest_node.dom_elt],
            #                                                nearest_node.label,
            #                                                round(node.nearest_dist, 5)))

        # 统计
        knn = [k.label for k in knn_list]
        return Counter(knn).most_common()[0][0]



    def find_nearest(self, tree, point):
        # 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
        result = namedtuple("Result_tuple", "nearest_node nearest_dist  nodes_visited")

        def travel(kd_node, target, max_dist):
            if kd_node is None or kd_node.visit:
                return result(None, float("inf"), 0)  # python中用float("inf")和float("-inf")表示正负无穷

            nodes_visited = 1

            s = kd_node.split  # 进行分割的维度
            pivot = kd_node.dom_elt  # 进行分割的“轴”

            if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
                nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
                further_node = kd_node.right  # 同时记录下右子树
            else:  # 目标离右子树更近
                nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
                further_node = kd_node.left

            temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

            nearest_node = temp1.nearest_node  # 以此叶结点作为“当前最近点”
            dist = temp1.nearest_dist  # 更新最近距离

            nodes_visited += temp1.nodes_visited

            if dist < max_dist:
                max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

            temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
            if max_dist < temp_dist:  # 判断超球体是否与超平面相交
                return result(nearest_node, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

            # ----------------------------------------------------------------------
            # 计算目标点与分割点的欧氏距离
            temp_dist = np.sqrt(np.sum(np.square(np.array(pivot) - np.array(target))))

            if temp_dist < dist:  # 如果“更近”
                nearest_node = kd_node  # 更新最近点
                dist = temp_dist  # 更新最近距离
                max_dist = dist  # 更新超球体半径

            # 检查另一个子结点对应的区域是否有更近的点
            temp2 = travel(further_node, target, max_dist)

            nodes_visited += temp2.nodes_visited
            if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
                nearest_node = temp2.nearest_node  # 更新最近点
                dist = temp2.nearest_dist  # 更新最近距离

            return result(nearest_node, dist, nodes_visited)

        return travel(tree.root, point, float("inf"))  # 从根节点开始递归

    def score(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)





def simpleTest():
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    label = [0, 0, 0, 1, 1, 1]
    kdtree = KNNKdTree()
    kdtree.fit(data, label)
    # kdtree.kdTree.preorder()
    predict_label = kdtree.predict([3, 4.5])
    print("predict label:", predict_label)

def largeTest():
    N = 400000
    startTime = time.time()
    data = random_points(2, N)
    label = [0] * (N // 2) + [1] * (N // 2)
    kdtree2 = KNNKdTree()
    kdtree2.fit(data, label)
    predict_label = kdtree2.predict([0.1, 0.5])  # 四十万个样本点中寻找离目标最近的点

    print("time: ", round(time.time() - startTime, 5), "s")
    print("predict label:", predict_label)


def main():
    simpleTest()
    largeTest()

if __name__ == "__main__":
    main()