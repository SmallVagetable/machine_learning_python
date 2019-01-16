
from em.gmm_penality import *
from sklearn import mixture

def getDataList():
    fileNameList = []
    dataList = ["amix1-est", "amix2-est", "golub-est"]
    for file in dataList:
        filePath = "./data/%s.dat" % (file)
        fileNameList.append(filePath)
    return fileNameList

def checkResult():
    fileNameList = getDataList()
    for fileName in fileNameList:
        X = np.loadtxt(fileName)
        searchK = 3
        penalty = [0, 1, 2]
        epoch = 2
        maxLogLikelihood = float('-inf')
        maxResult = None
        maxK = 0
        alpha = None
        bestP = 0
        for i in range(2, searchK):
            k = i
            for p in penalty:
                for j in range(epoch):
                    model1 = GMMPenality(k, penalty = p)
                    model1.fit(X)
                    if model1.loglike > maxLogLikelihood:
                        maxLogLikelihood = model1.loglike
                        maxResult, _ = model1.predict(X)
                        maxK = k
                        alpha = model1.alpha
                        bestP = p
                    alphaSorted = sorted(model1.alpha.tolist(), reverse=True)
                    print("fileName = %s, k = %s, penalty = %s alpha = %s, loglike = %s" % (fileName.split("/")[-1], k, p, [round(p[0], 5) for p in alphaSorted], round(model1.loglike, 5)))

        alpha, maxResult = changeLabel(alpha.reshape(1, -1).tolist()[0], maxResult)
        print("myself GMM alpha = %s, loglikelihood = %s, bestP = %s"%
              ([round(a, 5) for a in alpha], round(maxLogLikelihood, 5), bestP))

        # maxK = 3
        model2 = mixture.BayesianGaussianMixture(n_components=maxK,covariance_type='full')
        result2 = model2.fit_predict(X)
        sklearnAlpha, result2 = changeLabel(model2.weights_.tolist(), result2)

        result = np.sum(maxResult==result2)
        percent = np.mean(maxResult==result2)

        print("sklearn GMM alpha = %s, loglikelihood = %s"%
              ([round(a, 5) for a in sklearnAlpha], round(model2.lower_bound_, 5)))
        print("succ = %s/%s"%(result, len(result2)))
        print("succ = %s"%(percent))
        print(maxResult[:20])
        print(result2[:20])
        suffix = ["tst", "val"]
        for suf in suffix:
            newfileName = fileName.replace("-est", "-%s"%suf)
            newX = np.loadtxt(newfileName)
            maxResult, loglike = model1.predict(newX)
            print("fileName = %s, loglike = %s"%(newfileName.split("/")[-1],loglike))


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