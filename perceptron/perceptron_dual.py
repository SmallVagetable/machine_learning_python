import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils.plot import plot_decision_regions
from perceptron.perceptron_base import PerceptronBase


class PerceptronDual(PerceptronBase):
    """
    对偶形态感知机
    """
    def __init__(self, eta=0.1, n_iter=50):
        super(PerceptronDual, self).__init__(eta=eta, n_iter=n_iter)


    # 计算Gram Matrix
    def calculate_g_matrix(self, X):
        n_sample = X.shape[0]
        self.G_matrix = np.zeros((n_sample, n_sample))
        # 填充Gram Matrix
        for i in range(n_sample):
            for j in range(n_sample):
                self.G_matrix[i][j] = np.sum(X[i] * X[j])

    # 迭代的判定条件
    def judge(self, X, y, index):
        tmp = self.b
        n_sample = X.shape[0]
        for m in range(n_sample):
            tmp += self.alpha[m] * y[m] * self.G_matrix[index][m]

        return tmp * y[index]

    def fit(self, X, y):
        """
        对偶形态的感知机
        由于对偶形式中训练实例仅以内积的形式出现
        因此，若事先求出Gram Matrix，能大大减少计算量
        """
        # 读取数据集中含有的样本数,特征向量数
        n_samples, n_features = X.shape
        self.alpha, self.b = [0] * n_samples, 0
        self.w = np.zeros(n_features)
        # 计算Gram_Matrix
        self.calculate_g_matrix(X)

        i = 0
        while i < n_samples:
            if self.judge(X, y, i) <= 0:
                self.alpha[i] += self.eta
                self.b += self.eta * y[i]
                i = 0
            else:
                i += 1

        for j in range(n_samples):
            self.w += self.alpha[j] * X[j] * y[j]

        return self


def main():
    iris = load_iris()
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)
    ppn = PerceptronDual(eta=0.1, n_iter=10)
    ppn.fit(X_train, y_train)
    plot_decision_regions(ppn, X, y)


if __name__ == "__main__":
    main()
