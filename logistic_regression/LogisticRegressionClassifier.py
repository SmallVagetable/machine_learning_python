from sklearn.linear_model import LogisticRegression

from math import exp
from utils.data_generater import *


class LogisticRegressionClassifier(object):
    def __init__(self, max_iter=200, learning_rate=0.01):
        # 最大迭代次数
        self.max_iter = max_iter
        # 学习率
        self.learning_rate = learning_rate

    # sigmoid函数
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # 处理训练数据，增加一列，为了weight和bias合并处理
    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat


    def fit(self, X, y):
        data_mat = self.data_matrix(X)
        # self.weights包含了weight和bias合并处理
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                # 梯度下降迭代权重参数self.weights
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    # 计算准确度
    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = create_logistic_data()

    # 我们的LogisticRegression
    my_lr = LogisticRegressionClassifier()
    my_lr.fit(X_train, y_train)
    print("my LogisticRegression score", my_lr.score(X_test, y_test))

    # sklearn的LogisticRegression
    sklearn_lr = LogisticRegression(max_iter=200)
    sklearn_lr.fit(X_train, y_train)
    print("sklearn LogisticRegression score", sklearn_lr.score(X_test, y_test))
