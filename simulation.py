import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as LA

np.random.seed(0)

# read data
data = pd.read_csv("data.csv") 
returns = data.pct_change().dropna().values.T

# define parameters for experiment
n = 50
cov_true = np.cov(returns[0:n, :])
R = 0.99
all_m = [int(factor * n) for factor in np.linspace(0.5, 10, 20)]

# run experiment
num_runs = 50 
all_alpha = np.zeros((num_runs, len(all_m)))
for i in range(num_runs):
    for j, m in enumerate(all_m):
        X = multivariate_normal.rvs(mean=np.zeros((n, )), cov=cov_true, size=m).T
        X = X / np.linalg.norm(X, axis=0, keepdims=True)
        lmbda = (n / m) * (3 + 1 / R) * LA.norm(X @ X.T, 2) - 1
        alpha = lmbda / (lmbda + 1)
        all_alpha[i, j] = alpha


# plot
alpha_mean = np.mean(all_alpha, axis=0)
plt.figure(figsize=(6, 4))
plt.plot(all_m, alpha_mean, marker='o', linestyle='-', color='tab:blue', label=r'$\alpha$ mean')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel(r'$m$', fontsize=18)
plt.ylabel(r'$\alpha$', fontsize=18)
plt.ylim(0.8, 1.05)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label=r'$\alpha = 1$')
plt.tight_layout()
plt.savefig("alpha_mean_vs_m.pdf")
print("alpha mean:", alpha_mean)