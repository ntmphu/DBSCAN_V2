import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from multiprocessing import Pool
from tqdm import tqdm 
from util import *
import time




def run_parametric_dbscan(n, O_obs, minpts, eps, threshold, a, c):
    zk = - threshold
    list_zk = [zk]
    list_setofOutliers = []
    list_interval = []
    total_compute_z_time_1 = 0.0
    total_compute_z_time_2 = 0.0
    while zk < threshold:
        yzk = a*zk + c
        label_zk, neps_zk = dbscan_sk(a*zk + c, minpts, eps)
        setofOutliers =  [i for i in range(len(label_zk)) if label_zk[i] == -1]
        list_setofOutliers.append(setofOutliers)
        #if zk == -threshold:
            # label_test, neps_test = dbscan_sk(a*zk + c, minpts, eps)
            #setofOutliers =  [i for i in range(len(label_test)) if label_test[i] == -1]
            #print(label_test)
            #print(neps_test)

        start_time = time.time()
        intersection = compute_z_interval(n, O_obs, eps, neps_zk, a, c, zk)
        end_time = time.time()
        total_compute_z_time_1 += (end_time - start_time)

        start_time2 = time.time()
        intersection2 = compute_z_interval_test(n, O_obs, eps, neps_zk, a, c, zk)
        end_time2 = time.time()
        total_compute_z_time_2 += (end_time2 - start_time2)
        
        print("result", intersection)
        print("result_test", intersection2)
 

        for each_interval in intersection:
            if each_interval[0] <= zk <= each_interval[1]:

                next_zk = each_interval[1]
                list_interval.append([each_interval[0], each_interval[1]])
                break

        zk = next_zk + 0.0001        #[[zk;...], [zk+1,...]]
        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)
    print(f"Total time for compute_z_interval_1: {total_compute_z_time_1:.4f} seconds")
    print(f"Total time for compute_z_interval_2: {total_compute_z_time_2:.4f} seconds")
    

    return list_zk, list_interval, list_setofOutliers


def run_parametric(n):
  y = generate(n)
  eps = 0.2
  minpts = 5
  threshold = 20
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
  print(a.shape, c.shape)

  start_time = time.time()
  list_zk, list_z_interval, list_setofOutliers = run_parametric_dbscan(n, O, minpts, eps, threshold, a, c)
  run_parametric_dbscan_time = time.time() - start_time
  print("run_parametric_dbscan", run_parametric_dbscan_time )
  #print("list zk", len(list_zk), list_zk)
  #print("list O",  len(list_setofOutliers), list_setofOutliers)

  z_interval = []
  z_k = []

  for i in range(len(list_setofOutliers)):
    if np.array_equal(np.sort(list_setofOutliers[i]), np.sort(O)):
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
        if abs(sub) <= 0.001:
            new_z_interval[-1][1] = each_interval[1]
        else:
            new_z_interval.append(each_interval)

  for i in range (len(z_k)):
    if  z_interval[i][0]<= z_k[i] <= z_interval[i][1]:
      continue
    else:
      print("err", z_k[i], z_interval[i])
  #print(new_z_interval)
  etaTy = np.dot(etaT, y)
  mu = np.dot(etaT, np.zeros((n,1)))[0][0]
  cdf = pivot_with_specified_interval(new_z_interval, eta, etaTy[0][0], Sigma, mu)
  #print(z, new_z_interval)
  if cdf is None:
    return None

  selective_p_value = 2 * min(cdf, 1 - cdf)
  
  #print(selective_p_value)
 # if selective_p_value < 0.05:
  #  print("intersection, eta, z, etaT_Sigma_eta, mu", new_z_interval, z, etaT_Sigma_eta, mu)
  return selective_p_value

n = 100
np.random.seed(0)

for i in range (5):
    print(i)
    start_time = time.time()
    p = run_parametric(n)
    print(time.time() - start_time)
    print(p)



