# 1. 前言
前面几篇博文对EM算法和GMM模型进行了介绍，本文我们通过对GMM增加一个惩罚项。

# 2. 不带惩罚项的GMM
原始的GMM的密度函数是
$$
p(\boldsymbol{x}|\boldsymbol{\pi},\boldsymbol{\mu},\boldsymbol{\Sigma})=\sum_{k=1}^K\pi_k\mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)
$$
$$
\sum_{k=1}^K\pi_k=1
$$
其中$K$是高斯组件的个数，$[\pi_1,\pi_2,...,\pi_k]$是每个组件的权重。其中的$\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k$是组件$k$的均值和协方差矩阵。

log极大似然函数的公式是：
$$
L(\theta,\theta^{(j)})=\sum_{k=1}^Kn_k[log\pi_k-\frac{1}{2}(log(\boldsymbol{\Sigma_k})+\frac{{(x_i-\boldsymbol{\mu}_k})^2}{\boldsymbol{\Sigma}_k})]\;\;\;\;\;(1)
$$

这里有一个响应度的变量$\gamma_{ik}$，响应度$\gamma_{ik}$代表了第$i$个样本，在第$k$个组件上的响应程度。响应度的计算公式也很简单。
$$
\gamma_{ik}=\frac{\pi_k\mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}{\sum_{k=1}^K\pi_k\mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}
$$

通过$L(\theta, \theta^{j})$对$\mu_k$，$\Sigma_k$求偏倒等于0得到

$$
\mu_k=\frac{1}{n_k}\sum_{i=1}^N\gamma_{ik}x_i\;\;\;\;\;(2)
$$
$$
\Sigma_k=\frac{1}{n_k}\sum_{i=1}^N\gamma_{ik}(x_i-\mu_k)^2
$$
$$
\pi_k=\frac{n_k}{N}
$$
其中的$n_k=\sum_{i=1}^N\gamma_{ik}$。

到这里为止我们不带惩罚项的所有变量都计算出来了，只要一直循环E步M步，就能使得loglikelihood最大化。

# 3. 带惩罚项的GMM
在带penality的GMM中，我们假设协方差是一个对角矩阵，这样的话，我们计算高斯密度函数的时候，只需要把样本各个维度与对应的$\mu_k$和$\sigma_k$计算一维高斯分布，再相加即可。不需要通过多维高斯进行计算，也不需要协方差矩阵是半正定的要求。

我们给上面的(1)式加入一个惩罚项，
$$
\lambda\sum_{k=1}^K\sum_{j=1}^P\frac{|\mu_k-\bar{x}_j|}{s_j}
$$
其中的$P$是样本的维度。$\bar{x}_j$表示每个维度的平均值，$s_j$表示每个维度的标准差。这个penality是一个L1范式，对$\mu_k$进行约束。

加入penality后(1)变为
$$
L(\theta,\theta^{(j)})=\sum_{k=1}^Kn_k[log\pi_k-\frac{1}{2}(log(\boldsymbol{\Sigma_k})+\frac{{(x_i-\boldsymbol{\mu}_k})^2}{\boldsymbol{\Sigma}_k})] - \lambda\sum_{k=1}^K\sum_{j=1}^P\frac{|\mu_k-\bar{x}_j|}{s_j}
$$

这里需要注意的一点是，因为penality有一个绝对值，所以在对$\mu_k$求导的时候，需要分情况。于是(2)变成了
$$
\mu_k=\frac{1}{n_k}\sum_{i=1}^N\gamma_{ik}x_i
$$
$$
\mu_k= 
\left \{\begin{array}{cc}
\frac{1}{n_k}(\sum_{i=1}^N\gamma_{ik}x_i - \frac{\lambda\sigma^2}{s_j}), & \mu_k >= \bar{x}_j\\
\frac{1}{n_k}(\sum_{i=1}^N\gamma_{ik}x_i + \frac{\lambda\sigma^2}{s_j}), & \mu_k < \bar{x}_j
\end{array}\right.
$$

## 3.1 注意点

- 在带有penality的GMM中，如果从一开始迭代时，$\lambda>0$那这时loglikelihood很容易陷入一个局部最大值。如果前几个迭代我们先令$\lambda=0$,而后在令$\lambda>0$，这样能够寻找到一个比较好的最大值点。
- 由于在算EM的时候，很容易出现underflow活着overflow，这是我们可以通过一个近似公式来避开这个问题。
$$
log(\sum_hexp(a_h)) = m + log(\sum_hexp(a_h - m))\;\;\;m=max(a_h) 
$$
- 初始值很影响EM的聚类的结果，所以我们需要改变seed来多次运行程序，寻找导最好的EM结果。


# 4. 总结

本文对GMM模型进行了改良，加入了L1的penality项，使得$\mu_k$不会偏离$\bar{x}_j$太大，导致过拟合。下一篇博客通过代码，详细的展示这个过程。



