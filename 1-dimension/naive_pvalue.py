from util import *
import scipy.stats as stats

def naive(n):
  y = generate(n)
  eps = 0.2
  minpts = 5
  label, neps = dbscan_sk(y, minpts, eps)
  label = np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]

  if len(O) == 0 or len(O) == n:
    return None

  #test statistic

  j = np.random.choice(O)

  minusO = [i for i in range(label.shape[0]) if label[i] != -1]
  #print(len(O), len(minusO))
  eT_minusO = np.zeros((1, label.shape[0]))
  eT_minusO[:,minusO] = 1
  eT_mean_minusO = eT_minusO/(n - len(O))

  #test statistic
  eT_j = np.zeros((1, label.shape[0]))
  eT_j[:,j] = 1

  etaT = eT_j - eT_mean_minusO
  eta = np.transpose(etaT)
  etaTy = np.dot(etaT, y)

#calculate p-value
  Sigma = np.identity(n) #cov
  etaT_Sigma_eta=np.dot(np.dot(etaT, Sigma), eta)
  mu = 0
  std = np.sqrt(etaT_Sigma_eta)[0][0]
  cdf = np.array(stats.norm.cdf(etaTy[0][0],mu,std))

  if cdf is None:
    return None
  p_value = 2 * min(cdf, 1 - cdf)

  return p_value


def naive_TPR(args):
  n, delta = args
  y, true_outliers = generate_trueoutliers(n, delta)
  eps = 0.2
  minpts = 5
  label, neps = dbscan_sk(y, minpts, eps)
  label = np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]

  if len(O) == 0 or len(O) == n:
    return None, None

  #test statistic

  j = np.random.choice(O)

  minusO = [i for i in range(label.shape[0]) if label[i] != -1]
  #print(len(O), len(minusO))
  eT_minusO = np.zeros((1, label.shape[0]))
  eT_minusO[:,minusO] = 1
  eT_mean_minusO = eT_minusO/(n - len(O))

  #test statistic
  eT_j = np.zeros((1, label.shape[0]))
  eT_j[:,j] = 1

  etaT = eT_j - eT_mean_minusO
  eta = np.transpose(etaT)
  etaTy = np.dot(etaT, y)

#calculate p-value
  Sigma = np.identity(n) #cov
  etaT_Sigma_eta=np.dot(np.dot(etaT, Sigma), eta)
  mu = 0
  std = np.sqrt(etaT_Sigma_eta)[0][0]
  cdf = np.array(stats.norm.cdf(etaTy[0][0],mu,std))

  if cdf is None:
    return None, None
  p_value = 2 * min(cdf, 1 - cdf)
  if j in true_outliers:
    return p_value, True
  else:
    return p_value, False


