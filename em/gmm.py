from scipy.special import logsumexp
from utils.misc_utils import *



class GMM(object):
    def __init__(self, k, tol = 1e-3, reg_covar = 1e-7):
        self.K = k
        self.tol = tol
        self.reg_covar=reg_covar
        self.times = 100
        self.loglike = 0


    def fit(self, trainMat):
        self.X = trainMat
        self.N, self.D = trainMat.shape
        self.GMM_EM()

    # gmm入口
    def GMM_EM(self):
        self.scale_data()
        self.init_params()
        for i in range(self.times):
            log_prob_norm, self.gamma = self.e_step(self.X)
            self.mu, self.cov, self.alpha = self.m_step()
            newloglike = self.loglikelihood(log_prob_norm)
            # print(newloglike)
            if abs(newloglike - self.loglike) < self.tol:
                break
            self.loglike = newloglike


    #预测类别
    def predict(self, testMat):
        log_prob_norm, gamma = self.e_step(testMat)
        category = gamma.argmax(axis=1).flatten().tolist()[0]
        return np.array(category)


    #e步，估计gamma
    def e_step(self, data):
        gamma_log_prob = np.mat(np.zeros((self.N, self.K)))

        for k in range(self.K):
            gamma_log_prob[:, k] = log_weight_prob(data, self.alpha[k], self.mu[k], self.cov[k])

        log_prob_norm = logsumexp(gamma_log_prob, axis=1)
        log_gamma = gamma_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, np.exp(log_gamma)


    #m步，最大化loglikelihood
    def m_step(self):
        newmu = np.zeros([self.K, self.D])
        newcov = []
        newalpha = np.zeros(self.K)
        for k in range(self.K):
            Nk = np.sum(self.gamma[:, k])
            newmu[k, :] = np.dot(self.gamma[:, k].T, self.X) / Nk
            cov_k = self.compute_cov(k, Nk)
            newcov.append(cov_k)
            newalpha[k] = Nk / self.N

        newcov = np.array(newcov)
        return newmu, newcov, newalpha


    #计算cov，防止非正定矩阵reg_covar
    def compute_cov(self, k, Nk):
        diff = np.mat(self.X - self.mu[k])
        cov = np.array(diff.T * np.multiply(diff, self.gamma[:,k]) / Nk)
        cov.flat[::self.D + 1] += self.reg_covar
        return cov


    #数据预处理
    def scale_data(self):
        for d in range(self.D):
            max_ = self.X[:, d].max()
            min_ = self.X[:, d].min()
            self.X[:, d] = (self.X[:, d] - min_) / (max_ - min_)
        self.xj_mean = np.mean(self.X, axis=0)
        self.xj_s = np.sqrt(np.var(self.X, axis=0))


    #初始化参数
    def init_params(self):
        self.mu = np.random.rand(self.K, self.D)
        self.cov = np.array([np.eye(self.D)] * self.K) * 0.1
        self.alpha = np.array([1.0 / self.K] * self.K)


    #log近似算法，可以防止underflow，overflow
    def loglikelihood(self, log_prob_norm):
        return np.sum(log_prob_norm)


    # def loglikelihood(self):
    #     P = np.zeros([self.N, self.K])
    #     for k in range(self.K):
    #         P[:,k] = prob(self.X, self.mu[k], self.cov[k])
    #
    #     return np.sum(np.log(P.dot(self.alpha)))


