
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import sklearn
import numpy as np
from util import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
from parametric import run_parametric
# fetch dataset 

def run_parametric_dbscan(j, n, d, O_obs, minpts, eps, threshold, a, c):
  zk = - threshold
  list_zk = [zk]
  list_setofOutliers = []
  list_interval = []
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
    #print(zk, intersection)
       #[(interval1), (interval2)]
    #print(zk, intersection)
    #next_zk = zk
    for each_interval in intersection:
      if each_interval[0] <= zk <= each_interval[1]:
        #if each_interval[0] < list_zk[-1]:
         # print("err", each_interval[0] - list_zk[-1])
          #if each_interval[0] - list_zk[-1] == -float('inf'):
           # print(zk, each_interval)
        next_zk = each_interval[1]
        list_interval.append([each_interval[0], each_interval[1]])
        break

    zk = next_zk + 0.0001        #[[zk;...], [zk+1,...]]
    if zk < threshold:
      list_zk.append(zk)
    else:
      list_zk.append(threshold)


    #print(zk)
  return list_zk, list_interval, list_setofOutliers, list_sign

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

  list_zk, list_z_interval, list_setofOutliers, list_sign = run_parametric_dbscan(j, n, d, O, minpts, eps, threshold, a, c)
  #print("list zk", len(list_zk), list_zk)
  #print("list O",  len(list_setofOutliers), list_setofOutliers)

  z_interval = []
  z_k = []

  for i in range(len(list_setofOutliers)):
    if np.array_equal(np.sort(list_setofOutliers[i]), np.sort(O)) and np.array_equal(list_sign[i], S_obs):
          #print(i)
      z_interval.append([list_zk[i], list_zk[i + 1] - 0.00001])
      z_k.append(list_zk[i])
  #print(z_interval)
  #print(z_k)
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
  #print("z", z)
  #print(z)





  #print(intersection)

  z = etaTx[0][0]
  mu = 0

  cdf = pivot_with_specified_interval(new_z_interval, eta, z, Sigma, mu)
  if cdf is None:
      return None
  """
  if cdf is None or -1e-10 < cdf < 1e-10:
  
    print("----------------")
    print(f"O_obs = {O}")
    print(f"sign_obs = {S_obs.flatten()}")
    #print(f"neps_obs = {neps}")
    print(f"test statistic", z)
    print("parametric", new_z_interval)
    #print(list_zk)
    sioc_interval, sioc_sign = compute_z_interval(j, n, d, O, eps, neps, a, c, z, minusO, x)
    print("sioc", sioc_interval, sioc_sign.flatten())

    for i in range(len(list_setofOutliers)):
      if list_z_interval[i][0] <= z <= list_z_interval[i][1]:
        print(list_z_interval[i-1],list_z_interval[i], list_z_interval[i+1])
        print(list_setofOutliers[i-1], list_setofOutliers[i], list_setofOutliers[i+1])
        #print(list_neps[i])
        print(list_sign[i].flatten())

    if cdf is None:
      print("nonecdf")
      return None
    else:
      print("cdf = ", cdf)
      return None
    """
  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  return selective_p_value


def run_wrapper(args):
    X, minpts, eps = args    
    X_index = np.array(X.shape[0])
    subset_index = np.random.choice(X_index, size=150)
    X_subset = X[subset_index]
    return run_parametric(X_subset, minpts, eps)
if __name__ == '__main__':
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
    data = breast_cancer_wisconsin_diagnostic.data.original

    data = data.drop(['ID', 'Diagnosis'], axis='columns')

    scaler = StandardScaler()
    data = np.array(data)
    # Fit the scaler to the data
    newdata = scaler.fit_transform(data)
    X = np.array(newdata)
    X = X[:,10:15]
    
    max_iteration = 120
    minpts = 10
    eps = 2
    Alpha = 0.05
    list_p_value = []
    args = [(X, minpts, eps) for _ in range(max_iteration)]
    # Counter for false positives
    count = 0
    fpr = 0
    iter = 0
    with Pool() as pool:
        # Use tqdm with imap_unordered for real-time progress tracking
        for selective_p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
            if selective_p_value is not None:
                list_p_value.append(selective_p_value)
                if selective_p_value <= Alpha:
                    fpr += 1
                count += 1

    # Set up multiprocessing Pool and run computations in parallel


    # Calculate and print false positive rate
    print()
    print('False positive rate:', fpr / count, count)
    print(stats.kstest(list_p_value, stats.uniform(loc=0.0, scale=1.0).cdf))

    # Plot histogram of p-values
    plt.hist(list_p_value)
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.title('Selective P-Value Distribution')
    plt.show()

