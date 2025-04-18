import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from multiprocessing import Pool
from tqdm import tqdm 
from util import *




def run_parametric_dbscan(n, O_obs, minpts, eps, threshold, a, c):
  zk = - threshold
  list_zk = [zk]
  list_setofOutliers = []
  list_interval = []
  while zk < threshold:
    yzk = a*zk + c
    label_zk, neps_zk = dbscan_sk(a*zk + c, minpts, eps)
    setofOutliers =  [i for i in range(len(label_zk)) if label_zk[i] == -1]
    list_setofOutliers.append(setofOutliers)
  
    intersection = compute_z_interval(n, O_obs, eps, neps_zk, a, c, zk)
    
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


    #print(zk)
  return list_zk, list_interval, list_setofOutliers


def run_parametric(n, gen_func):
  y, Sigma = gen_func(n)
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
  #Sigma = np.identity(n) #cov
  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  threshold = 20*etaT_Sigma_eta[0][0]
  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(n) - np.dot(a, eta.T), y)


  list_zk, list_z_interval, list_setofOutliers = run_parametric_dbscan(n, O, minpts, eps, threshold, a, c)
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
  #print("z interval", len(z_interval), z_interval)
  #print("new z interval", len(new_z_interval), new_z_interval)
  #print("z interval", len(z_interval), z_interval)
  #print("zk", len(z_k), z_k)
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

def run_parametric_estimatesigma(n):
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

  #compute a, c: y = az + c((
  variance = np.var(y)
  Sigma = variance * np.identity(n) #cov
  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(n) - np.dot(a, eta.T), y)


  list_zk, list_z_interval, list_setofOutliers = run_parametric_dbscan(n, O, minpts, eps, threshold, a, c)
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
  #print("z interval", len(z_interval), z_interval)
  #print("new z interval", len(new_z_interval), new_z_interval)
  #print("z interval", len(z_interval), z_interval)
  #print("zk", len(z_k), z_k)
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

def run_wrapper(args):
    n, func = args
    return run_parametric(n, func)


if __name__ == "__main__":
    max_iteration = 300
    Alpha = 0.05
    count = 0
    n = 50
    args = [(n, generate_skewnorm) for _ in range(max_iteration)]
    
    # Khởi tạo Pool và chạy các tính toán song song
    with Pool(initializer=np.random.seed) as pool:
        # Use imap_unordered to update tqdm as each result completes
        results = []
        for result in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
            results.append(result)
    # Lọc các giá trị None và tính toán
    list_p_value = [p_value for p_value in results if p_value is not None]

    # Đếm số lần false positive
    count = sum(1 for p_value in list_p_value if p_value <= Alpha)

    # Tính và in False positive rate
    print('\nFalse positive rate:', count / len(list_p_value), len(list_p_value))

    # Kiểm định thống kê
    print(stats.kstest(list_p_value, stats.uniform(loc=0.0, scale=1.0).cdf))

    # Hiển thị histogram
    plt.hist(list_p_value)

    plt.show()



