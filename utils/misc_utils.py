import numpy as np
import numbers
from scipy.stats import multivariate_normal

def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2), axis=1))


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def sortLabel(label):
    label = np.array(label)
    labelOld = []
    labelNum = len(list(set(label)))
    for i in label:
        if i not in labelOld:
            labelOld.append(i)
        if len(labelOld) == labelNum:
            break

    labelNew = sorted(labelOld)
    for i, old in enumerate(labelOld):
        label[label == old] = labelNew[i] + 10000
    return label - 10000

def prob(x, mu, cov):
    norm = multivariate_normal(mean=mu, cov=cov)
    return norm.pdf(x)

def log_prob(x, mu, cov):
    norm = multivariate_normal(mean=mu, cov=cov)
    return norm.logpdf(x)


def log_weight_prob(x, alpha, mu, cov):
    N = x.shape[0]
    return np.mat(np.log(alpha) + log_prob(x, mu, cov)).reshape([N, 1])