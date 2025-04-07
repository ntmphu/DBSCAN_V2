from util import *
import scipy.stats as stats



def run_sioc(X, minpts, eps):
  n = X.shape[0]
  d = X.shape[1]

  label, neps = dbscan_sk(X, minpts, eps)
  label = np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]
    #print(O)
  if len(O) == 0 or len(O) == n:
    return None
  #test statistic

  j = np.random.choice(O)
  #print(X[j])
  minusO = [i for i in range(label.shape[0]) if label[i] != -1]
  
  covX = np.cov(X[minusO,:], rowvar=False)  # Covariance matrix of columns


  # Construct Cov(vec(X)) using Kronecker product
  Sigma = np.kron(np.eye(n), covX)
  #print(len(O), len(minusO))
  eT_minusO = np.zeros((1, label.shape[0]))
  eT_minusO[:,minusO] = 1
  x = vec(X)
  #print(X)
  #print(x)
  I_d = np.identity(d)
  eT_mean_minusO = np.kron(I_d, eT_minusO)/(n - len(O))
  #print(np.dot(eT_mean_minusO, x), np.mean(X[minusO], axis = 0))

  e_j = np.zeros((1, n))
  e_j[:,j] = 1
  temp = np.kron(I_d, e_j) - eT_mean_minusO
  Xj_meanXminusO = np.dot(temp, x)
  #print(Xj_meanXminusO, X[j] - np.mean(X[minusO], axis = 0))

  S = np.sign(Xj_meanXminusO)
 

  etaT = np.dot(S.T, temp)/d
  eta = np.transpose(etaT)
  #print("eta", eta.shape)
  #print("x", x.shape)

  etaTx = np.dot(etaT, x)

  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  #print(etaT_Sigma_eta)


  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(int(n*d)) - np.dot(a,eta.T), x)
  z = etaTx
  #print("z", z)
  #print(z)
  intersection, _ = compute_z_interval(j, n, d, O, eps, neps, a, c, z, minusO, x)

  #print(interval)
  #print(intersection)

  z = etaTx[0][0]
  mu = 0

  cdf = pivot_with_specified_interval(intersection, eta, z, Sigma, mu)
  if cdf is None:
    return None
  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  
  return selective_p_value



