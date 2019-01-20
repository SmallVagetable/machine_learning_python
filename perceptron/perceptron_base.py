import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils.plot import plot_decision_regions


class PerceptronBase(object):
    """
    原始形态感知机
    """
    def __init__(self, eta=0.1, n_iter=50):
        # 学习率
        self.eta = eta
        # 迭代次数
        self.n_iter = n_iter

    def fit(self, X, y):
        # 初始化参数w，b
        self.w = np.zeros(X.shape[1])
        self.b = 0
        # 记录所有error
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            if errors == 0:
                break
            self.errors_.append(errors)

        return self

    def sign(self, xi):
        return np.dot(xi, self.w) + self.b

    def predict(self, xi):
        return np.where(self.sign(xi) <= 0.0, -1, 1)


def main():
    iris = load_iris()
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)
    ppn = PerceptronBase(eta=0.1, n_iter=10)
    ppn.fit(X_train, y_train)
    plot_decision_regions(ppn, X, y)


if __name__ == "__main__":
    main()
