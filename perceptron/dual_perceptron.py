import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

a, b, G_matrix = 0, 0, 0

# 计算Gram Matrix
def calculate_g_matrix(data):
    global G_matrix
    G_matrix = np.zeros((data.shape[0], data.shape[0]))
    # 填充Gram Matrix
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            G_matrix[i][j] = np.sum(data[i, 0:-1] * data[j, 0:-1])

# 迭代的判定条件
def judge(data, y, index):
    global a, b
    tmp = 0
    for m in range(data.shape[0]):
        tmp += a[m] * data[m, -1] * G_matrix[index][m]

    return (tmp + b) * y


def dual_perceptron(data):
    """
    对偶形态的感知机
    由于对偶形式中训练实例仅以内积的形式出现
    因此，若事先求出Gram Matrix，能大大减少计算量
    :param data:训练数据集;ndarray object
    :return:w,b
    """
    global a, b, G_matrix

    # 计算Gram_Matrix
    calculate_g_matrix(data)

    # 读取数据集中含有的样本数
    num_samples = data.shape[0]
    # 读取数据集中特征向量的个数
    num_features = data.shape[1] - 1
    # 初始化a,b
    a, b = [0] * num_samples, 0
    # 初始化weight
    w = np.zeros((1, num_features))

    i = 0
    while i < num_samples:
        if judge(data, data[i, -1], i) <= 0:
            a[i] += 1
            b += data[i, -1]
            i = 0
        else:
            i += 1

    for j in range(num_samples):
        w += a[j] * data[j, 0:-1] * data[j, -1]

    return w, b