from util import *
import scipy.stats as stats

def run_naive(n, d, minpts, eps):
  X, u, Sigma = generate(n, d)

  label, neps = dbscan_sk(X, minpts, eps)
  label =np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]

  if len(O) == 0 or len(O) == n:
    return None
  #test statistic

  j = np.random.choice(O)
  minusO = [i for i in range(label.shape[0]) if label[i] != -1]

  eT_minusO = np.zeros((1, label.shape[0]))
  eT_minusO[:,minusO] = 1
  x = vec(X)

  I_d = np.identity(d)
  eT_mean_minusO = np.kron(I_d, eT_minusO)/(n - len(O))

  e_j = np.zeros((1, n))
  e_j[:,j] = 1
  temp = np.kron(I_d, e_j) - eT_mean_minusO
  Xj_meanXminusO = np.dot(temp, x)
  S = np.sign(Xj_meanXminusO)

  etaT = np.dot(S.T, temp)
  eta = np.transpose(etaT)
  etaTx = np.dot(etaT, x)[0][0]
  mu = np.dot(etaT, u)[0][0]
#calculate p-value
  etaT_Sigma_eta=np.dot(np.dot(etaT, Sigma), eta)
  std = np.sqrt(etaT_Sigma_eta)[0][0]
  cdf = np.array(stats.norm.cdf(etaTx,mu,std))

  if cdf is None:
    return None
  p_value = 2 * min(cdf, 1 - cdf)

  return p_value

def run_naive_TPR(n, d, delta, minpts, eps):
  X, u, Sigma, true_outliers = generate_trueoutliers(n, d, delta)

  label, neps = dbscan_sk(X, minpts, eps)
  label = np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]
    #print(O)
  if len(O) == 0 or len(O) == n:
    return None, None
  #test statistic

  j = np.random.choice(O)
  #print(X[j])
  minusO = [i for i in range(label.shape[0]) if label[i] != -1]
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
  #print(S)
  B = np.multiply(S, temp)


  etaT = np.dot(S.T, temp)
  eta = np.transpose(etaT)
  #print("eta", eta.shape)
  #print("x", x.shape)

  etaTx = np.dot(etaT, x)[0][0]

#calculate p-value
  etaT_Sigma_eta=np.dot(np.dot(etaT, Sigma), eta)
  mu = np.dot(etaT, u)[0][0]

  std = np.sqrt(etaT_Sigma_eta)[0][0]
  cdf = np.array(stats.norm.cdf(etaTx,mu,std))
  if cdf is None:
    return None,  None
  p_value = 2 * min(cdf, 1 - cdf)
  if j in true_outliers:
    return p_value, True
  else:
    return p_value, False