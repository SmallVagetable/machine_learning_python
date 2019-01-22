import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm

class GaussianNaiveBayes(object):
    def __init__(self):
        self.model = None

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        return norm.pdf(x, loc=mean, scale=stdev)

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(np.mean(X), np.std(X)) for X in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return self

    # 计算概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

if __name__ == "__main__":
    iris = load_iris()
    X,y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(len(X_train))
    print(len(X_test))
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    print(model.predict([4.4, 3.2, 1.3, 0.2]))
    print(model.score(X_test, y_test))