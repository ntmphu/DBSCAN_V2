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

def run_parametric(n, d, minpts, eps):

  X, u, Sigma = generate(n,d)

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
  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  return selective_p_value

def run_parametric_TPR(n, d, delta, minpts, eps):
  X, u, Sigma, true_outliers = generate_trueoutliers(n,d, delta)
  
  threshold = 20
  
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
    return None, None
  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  if j in true_outliers:
    return selective_p_value, True
  else:
    return selective_p_value, False


def run_wrapper(args):
    n, d, minpts, eps = args
    return run_parametric(n, d, minpts, eps)


if __name__ == '__main__':

    max_iteration = 150
    n = 50
    d = 10
    minpts = 10
    eps = 3
    Alpha = 0.05
    list_p_value = []
    

    
    args = [(n, d, minpts, eps) for _ in range(max_iteration)]

    # Counter for false positives
    count = 0
    fpr = 0
    count_zero = 0

    # Set up multiprocessing Pool and run computations in parallel
    with Pool() as pool:
        # Use tqdm with imap_unordered for real-time progress tracking
        for selective_p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
            if selective_p_value is not None:
                list_p_value.append(selective_p_value)
                if selective_p_value <= Alpha:
                    fpr += 1
                    
                count += 1

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
