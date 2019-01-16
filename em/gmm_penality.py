import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp


class GMMPenality(object):

    def __init__(self, K, tol = 1e-3, penalty = 1):
        self.K = K
        self.tol = tol
        self.times = 80
        self.penalty = penalty
        self.beginPenaltyTime = 10


    def fit(self, train):
        # self.X = self.scale_data(train)
        self.X = train
        self.GMM_EM()


    def init_paras(self):
        self.N, self.D = self.X.shape
        self.means = np.mean(self.X, axis=0)
        self.std = np.sqrt(np.var(self.X, axis=0))

        self.mu = np.random.rand(self.K, self.D)
        self.sigma = np.random.rand(self.K, self.D)
        self.alpha = np.array([1.0 / self.K] * self.K)

        self.gamma = np.random.rand(self.N, self.K)
        self.loglike = 0

    def scale_data(self, X):
        for i in range(X.shape[1]):
            max_ = X[:, i].max()
            min_ = X[:, i].min()
            X[:, i] = (X[:, i] - min_) / (max_ - min_)
        return X

    def GMM_EM(self):
        self.init_paras()
        for i in range(self.times):
            #m step
            self.m_step(i)
            # e step
            logGammaNorm, self.gamma= self.e_step(self.X)
            #loglikelihood
            loglike = self.logLikelihood(logGammaNorm)
            #penalty
            pen = 0
            if i >= self.beginPenaltyTime:
                for j in range(self.D):
                    pen += self.penalty * np.sum(abs(self.mu[:,j] - self.means[j])) / self.std[j]

            # print("step = %s, alpha = %s, loglike = %s"%(i, [round(p[0], 5) for p in self.alpha.tolist()], round(loglike - pen, 5)))
            # if abs(self.loglike - loglike) < self.tol:
            #     break
            # else:

            self.loglike = loglike - pen

    def e_step(self, data):
        N, D = data.shape
        gamma = np.random.rand(N, self.K)
        for k in range(self.K):
            gamma[:, k] = np.log(self.alpha[k]) + self.log_prob(data, self.mu[k,], self.sigma[k, :])

        logGammaNorm = logsumexp(gamma, axis=1)
        gamma = np.exp(gamma - logGammaNorm[:, np.newaxis])
        return logGammaNorm, gamma


    def m_step(self, step):
        gammaNorm = np.array(np.sum(self.gamma, axis=0)).reshape(self.K, 1)
        self.alpha = gammaNorm / np.sum(gammaNorm)
        for k in range(self.K):
            Nk = gammaNorm[k]
            if Nk == 0:
                continue
            for j in range(self.D):
                if step >= self.beginPenaltyTime:
                    # 算出penality的偏移量shift，通过当前维度的mu和样本均值比较，确定shift的符号，相当于把lasso的绝对值拆开了
                    shift = np.square(self.sigma[k, j]) * self.penalty / (self.std[j] * Nk)
                    if self.mu[k, j] >= self.means[j]:
                        shift = shift
                    else:
                        shift = -shift
                else:
                    shift = 0
                self.mu[k, j] = np.dot(self.gamma[:, k].T, self.X[:, j]) / Nk - shift
                self.sigma[k, j] = np.sqrt(np.sum(np.multiply(self.gamma[:, k], np.square(self.X[:, j] - self.mu[k, j]))) / Nk)


    def predict(self, test):
        logGammaNorm, gamma = self.e_step(test)
        category = gamma.argmax(axis=1).flatten().tolist()
        return np.array(category),self.logLikelihood(logGammaNorm)


    #计算极大似然,通过np.sum所有的logGammaNorm
    def logLikelihood(self, logNorm):
        return np.sum(logNorm)


    #计算高斯密度概率函数，样本的高斯概率密度函数，其实就是每个一维mu,sigma的高斯的和
    def log_prob(self, X, mu, sigma):
        N, D = X.shape
        logRes = np.zeros(N)
        for i in range(N):
            a = norm.logpdf(X[i,:], loc=mu, scale=sigma)
            logRes[i] = np.sum(a)
        return logRes