from util import *

def run_sioc(n):
  y = generate(n)
  eps = 0.2
  minpts = 5
  label, neps = dbscan_sk(y, minpts, eps)
  label = np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]

  if len(O) == 0 or len(O) == n:
    return None

  #compute etaj
  j = np.random.choice(O)

  minusO = [i for i in range(label.shape[0]) if label[i] != -1]

  eT_minusO = np.zeros((1, label.shape[0]))
  eT_minusO[:,minusO] = 1
  eT_mean_minusO = eT_minusO/(n - len(O))

  eT_j = np.zeros((1, label.shape[0]))
  eT_j[:,j] = 1

  etaT = eT_j - eT_mean_minusO
  eta = np.transpose(etaT)

  #compute a, c: y = az + c
  Sigma = np.identity(n) #cov
  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(n) - np.dot(a, eta.T), y)
  z = np.dot(etaT, y)[0][0]
  intersection = compute_z_interval(n, O, eps, neps, a, c, z)
  mu = np.dot(etaT, np.zeros((n,1)))[0][0]
  cdf = pivot_with_specified_interval(intersection, eta, z, Sigma, mu)
  #print(z, new_z_interval)
  if cdf is None:
    return None

  selective_p_value = 2 * min(cdf, 1 - cdf)
  return selective_p_value

def run_sioc_TPR(args):
  n, delta = args
  y, true_outliers = generate_trueoutliers(n, delta)
  eps = 0.2
  minpts = 5
  label, neps = dbscan_sk(y, minpts, eps)
  label = np.array(label)
  O = [i for i in range(label.shape[0]) if label[i] == -1]

  if len(O) == 0 or len(O) == n:
    return None, None

  #compute etaj
  j = np.random.choice(O)

  minusO = [i for i in range(label.shape[0]) if label[i] != -1]

  eT_minusO = np.zeros((1, label.shape[0]))
  eT_minusO[:,minusO] = 1
  eT_mean_minusO = eT_minusO/(n - len(O))

  eT_j = np.zeros((1, label.shape[0]))
  eT_j[:,j] = 1

  etaT = eT_j - eT_mean_minusO
  eta = np.transpose(etaT)

  #compute a, c: y = az + c
  Sigma = np.identity(n) #cov
  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(n) - np.dot(a, eta.T), y)
  z = np.dot(etaT, y)[0][0]
  intersection = compute_z_interval(n, O, eps, neps, a, c, z)
  mu = np.dot(etaT, np.zeros((n,1)))[0][0]
  cdf = pivot_with_specified_interval(intersection, eta, z, Sigma, mu)
  #print(z, new_z_interval)
  if cdf is None:
    return None, None

  selective_p_value = 2 * min(cdf, 1 - cdf)
  if j in true_outliers:
    return selective_p_value, True
  else:
    return selective_p_value, False