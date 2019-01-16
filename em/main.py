from em.gmm import *
from sklearn import mixture



def checkResult():
    X = np.loadtxt("./data/amix1-est.dat")
    searchK = 4
    epoch = 5
    maxLogLikelihood = 0
    maxResult = None
    maxK = 0
    alpha = None
    for i in range(2, searchK):
        k = i
        for j in range(epoch):
            model1 = GMM(k)
            model1.fit(X)
            if model1.loglike > maxLogLikelihood:
                maxLogLikelihood = model1.loglike
                maxResult = model1.predict(X)
                maxK = k
                alpha = model1.alpha

    alpha, maxResult = changeLabel(alpha, maxResult)
    print("my gmm k = %s, alpha = %s, maxloglike = %s"%(maxK,[round(a, 5) for a in alpha],maxLogLikelihood))


    model2 = mixture.BayesianGaussianMixture(n_components=maxK,covariance_type='full')
    result2 = model2.fit_predict(X)
    alpha2, result2 = changeLabel(model2.weights_.tolist(), result2)

    result = np.sum(maxResult==result2)
    percent = np.mean(maxResult==result2)
    print("sklearn gmm k = %s, alpha = %s, maxloglike = %s"%(maxK,[round(a, 5) for a in alpha2],model2.lower_bound_))

    print("succ = %s/%s"%(result, len(result2)))
    print("succ = %s"%(percent))

    print(maxResult[:100])
    print(result2[:100])


def changeLabel(alpha, predict):
    alphaSorted = sorted(alpha, reverse=True)
    labelOld = []
    for i in predict:
        if i not in labelOld:
            labelOld.append(i)
        if len(labelOld) == len(alpha):
            break
    labelNew = sorted(labelOld)
    for i, old in enumerate(labelOld):
        predict[predict == old] = labelNew[i] + 100
    return alphaSorted, predict - 100


if __name__ == "__main__":
    checkResult()

