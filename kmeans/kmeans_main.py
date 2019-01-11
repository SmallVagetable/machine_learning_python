from kmeans.kmeans_base import KMeansBase
from sklearn import datasets

if __name__ == "__main__":




    iris = datasets.load_iris()
    km = KMeansBase(3)
    for k in km.fit(iris.data):
        print(k)