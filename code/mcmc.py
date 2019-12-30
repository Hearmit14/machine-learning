# markov
import numpy as np
matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=float)
vector1 = np.matrix([[0.3,0.4,0.3]], dtype=float)
for i in range(100):
    vector1 = vector1*matrix
    print("Current round:" , i+1)
    print(vector1)

# mh
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
%matplotlib inline

def norm_dist_prob(theta):
    y = norm.pdf(theta, loc=3, scale=2)
    return y

T = 100000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T-1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
    alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))
    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        # pi[t] = pi[t - 1]
        t = t -1

plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
num_bins = 1000
plt.hist(pi, num_bins, density=1, facecolor='red', alpha=0.7)
plt.show()

# gibbs
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
samplesource = multivariate_normal(mean=[5,-1], cov=[[1,1],[1,4]])

def p_ygivenx(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt((1 - rho ** 2) * (s2**2))))

def p_xgiveny(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt((1 - rho ** 2) * (s1**2))))

N = 50000
K = 20
x_res = []
y_res = []
z_res = []
m1 = 5
m2 = -1
s1 = 1
s2 = 2

rho = 0.5
y = m2

for i in range(N):
    for j in range(K):
        x = p_xgiveny(y, m1, m2, s1, s2)
        y = p_ygivenx(x, m1, m2, s1, s2)
        z = samplesource.pdf([x,y])
    x_res.append(x)
    y_res.append(y)
    z_res.append(z)

len(z_res)

num_bins = 50
plt.hist(x_res, num_bins, normed=1, facecolor='green', alpha=0.5)
plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.title('Histogram')
plt.show()

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res,marker='o')
plt.show()


# 你好，这里代码里的用法的确和算法描述稍有不同：

# 在代码里外层循环是得到Ｎ个样本，对于每个样本，我们要在内层循环中分别进行马尔科夫状态转移Ｋ次得到对应的样本。这样的好处是得到的样本独立性好一点，毕竟我们的条件概率计算量不大。

# 在算法里描述的的确如你所说，先进行若干次马尔科夫状态转移，到了我们认为已经收敛到目标分布时，就进行连续的采样，得到Ｎ个样本。
# 这样的好处是采样的计算量小很多，尤其是对于复杂分布的计算，一次马尔科夫过程后就可以得到所有想要个数的样本，但是样本之间的独立性会稍差。