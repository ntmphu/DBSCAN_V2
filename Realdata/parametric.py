import numpy as np
from scipy.stats import matrix_normal
from util import *
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
def run_parametric_dbscan(j, n, d, O_obs, minpts, eps, threshold, a, c):
  zk = - threshold
  list_zk = [zk]
  list_setofOutliers = []
  list_sign = []
  while zk < threshold:
    x_zk = a*zk + c
    X_zk = unvec(x_zk, n, d)
    label_zk, neps_zk = dbscan_sk(X_zk, minpts, eps)
    setofOutliers =  [i for i in range(len(label_zk)) if label_zk[i] == -1]
    minusO_zk = [i for i in range(len(label_zk)) if label_zk[i] != -1]
    list_setofOutliers.append(setofOutliers)
    #if zk == -threshold:
     # label_test, neps_test = dbscan_sk(a*zk + c, minpts, eps)
      #setofOutliers =  [i for i in range(len(label_test)) if label_test[i] == -1]
      #print(label_test)
      #print(neps_test)

    intersection, S = compute_z_interval(j, n, d, O_obs, eps, neps_zk, a, c, zk, minusO_zk, x_zk)
    list_sign.append(S)
    for each_interval in intersection:
      if each_interval[0] <= zk <= each_interval[1]:
        next_zk = each_interval[1]
        break

    zk = next_zk + 0.0001        
    if zk < threshold:
      list_zk.append(zk)
    else:
      list_zk.append(threshold)

  return list_zk, list_setofOutliers, list_sign

def run_parametric(X, minpts, eps):

  n = X.shape[0]
  d = X.shape[1]

  threshold = 20
  
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

  S_obs = np.sign(Xj_meanXminusO)
  #print(S)
  

  etaT = np.dot(S_obs.T, temp)/d
  eta = np.transpose(etaT)
  #print("eta", eta.shape)
  #print("x", x.shape)
  etaTx = np.dot(etaT, x)
  #print(etaTx)
  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  #print(etaT_Sigma_eta)
  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(int(n*d)) - np.dot(a,eta.T), x)

  list_zk, list_setofOutliers, list_sign = run_parametric_dbscan(j, n, d, O, minpts, eps, threshold, a, c)
  #print("list zk", len(list_zk), list_zk)
  #print("list O",  len(list_setofOutliers), list_setofOutliers)

  z_interval = []
  for i in range(len(list_setofOutliers)):
    if np.array_equal(np.sort(list_setofOutliers[i]), np.sort(O)) and np.array_equal(list_sign[i], S_obs):
          #print(i)
      z_interval.append([list_zk[i], list_zk[i + 1] - 0.0001])
      
  new_z_interval = []
  for each_interval in z_interval:
    if len(new_z_interval) == 0:
        new_z_interval.append(each_interval)
    else:
        sub = each_interval[0] - new_z_interval[-1][1]
        if abs(sub) <= 0.0001:
            new_z_interval[-1][1] = each_interval[1]
        else:
            new_z_interval.append(each_interval)

  z = etaTx[0][0]
  mu = 0

  cdf = pivot_with_specified_interval(new_z_interval, eta, z, Sigma, mu)
  if cdf is None:
      return None

  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  
  return selective_p_value
