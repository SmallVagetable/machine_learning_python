import numpy as np
from random import random


def makeRandomPoint(num, dim, upper):
    return np.random.normal(loc=upper, size=[num, dim])


# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]


# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]