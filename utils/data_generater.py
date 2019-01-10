import numpy as np

def makeRandomPoint(num, dim, upper):
    return np.random.normal(loc=upper, size=[num, dim])