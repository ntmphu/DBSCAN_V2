import numpy as np
from scipy.stats import matrix_normal
from util import *
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats


def run_sioc(n, d, minpts, eps, rho):
  X, u, Sigma, _ = generate_correlated_data(n, d, 0, rho)

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

  S = np.sign(Xj_meanXminusO)
 

  etaT = np.dot(S.T, temp)
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
  mu = np.dot(etaT,u)[0][0]

  cdf = pivot_with_specified_interval(intersection, eta, z, Sigma, mu)
  if cdf is None:
    return None
  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  if selective_p_value <0:
    print("intersection, eta, z, etaT_Sigma_eta, mu",intersection, z, etaT_Sigma_eta, mu)
  return selective_p_value





def run_sioc_TPR(n, d, delta, minpts, eps, rho):
  
  X, u, Sigma, true_outliers = generate_correlated_data(n, d, delta, rho)
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

  etaTx = np.dot(etaT, x)

  etaT_Sigma_eta=np.dot(np.dot(eta.T, Sigma), eta)
  #print(etaT_Sigma_eta)


  a = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
  c = np.dot(np.identity(int(n*d)) - np.dot(a,eta.T), x)
  z = etaTx

  intersection, _ = compute_z_interval(j, n, d, O, eps, neps, a, c, z, minusO, x)
  #print(interval
  

  z = etaTx[0][0]
  mu = np.dot(etaT, u)[0][0]

  cdf = pivot_with_specified_interval(intersection, eta, z, Sigma, mu)
  if cdf is None:
    return None, None
  selective_p_value = 2 * min(cdf, 1 - cdf)
  #print(selective_p_value)
  if j in true_outliers:
    return selective_p_value, True
  else:
    return selective_p_value, False

def run_wrapper(args):
    n, d, minpts, eps, rho = args
    return run_sioc(n, d, minpts, eps, rho)


if __name__ == '__main__':

    max_iteration = 500
    n = 200
    d = 10
    minpts = 10
    eps = 3
    Alpha = 0.05
    rho = 0.5
    list_p_value = []
    
    
    args = [(n, d, minpts, eps, rho) for _ in range(max_iteration)]

    # Counter for false positives
    count = 0
    fpr = 0

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


