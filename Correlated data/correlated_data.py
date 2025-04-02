import numpy as np
import scipy.stats as stats
def vec(A):
  vec = A.reshape(-1, order='F')
  return vec.reshape(-1,1)
def generate_trueoutliers(n, d, delta):
  M = np.zeros((n, d))
  u = vec(M)
  U = np.identity(n)
  V = np.identity(d)

  M_index = np.array(range(n))
  true_outliers = np.random.choice(M_index, size=(n//5), replace=False)
  M[true_outliers] += delta
  X = M + stats.matrix_normal.rvs(mean=M, rowcov=U, colcov=V)
  Sigma = np.kron(V,U)
  return X, u, Sigma, true_outliers


def generate(n, d, no_outlier_s, rho):
    mu_Xs = np.zeros(d)
    true_Xs = np.array([mu_Xs for _ in range(n)])
    
    ys = np.zeros(n)

    # Generate outlier
    idx_s = np.random.randint(n, size=no_outlier_s)
    idx_s = np.unique(idx_s)
    true_Xs[idx_s] = true_Xs[idx_s] + 2
    ys[idx_s] = 1

    cov_matrix = np.array([[rho**abs(i-j) for j in range(d)] for i in range(d)])

    Xs_obs = true_Xs + np.random.multivariate_normal(mu_Xs, cov_matrix, size=n)

    return Xs_obs, true_Xs,  ys, cov_matrix
def generate_correlated_data(n, d, delta, rho):
  M = np.zeros((n, d))
  u = vec(M)
  U = np.identity(n)
  V = np.array([[rho**abs(i-j) for j in range(d)] for i in range(d)])

  M_index = np.array(range(n))
  true_outliers = np.random.choice(M_index, size=5, replace=False)
  M[true_outliers] += delta
  
  
  X = M + stats.matrix_normal.rvs(mean=np.zeros((n,d)), rowcov=U, colcov=V)
  Sigma = np.kron(V,U)
  return X, u, Sigma, true_outliers
n = 5
d = 2
rho = 0.25
delta = 2
X, u, Sigma, true_outliers = generate_correlated_data(n, d, delta, rho)
