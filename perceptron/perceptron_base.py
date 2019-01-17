import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils.plot import plot_decision_regions


class PerceptronBase(object):
    """
    原始形态感知机
    """
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta  # 学习率
        self.n_iter = n_iter  # 迭代次数

    def fit(self, X, y):
        """
        拟合函数，使用训练集来拟合模型
        :param X:training sets
        :param y:training labels
        :return:self
        """
        # X's each col represent a feature
        # initialization wb(weight plus bias)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        # the main process of fitting
        self.errors_ = []  # store the errors for each iteration
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
        """
        计算净输入
        :param xi:
        :return:净输入
        """
        return np.dot(xi, self.w) + self.b

    def predict(self, xi):
        """
        计算预测值
        :param xi:
        :return:-1 or 1
        """
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
