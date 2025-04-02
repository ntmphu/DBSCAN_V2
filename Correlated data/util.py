import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpmath import mp
import scipy.stats as stats
from scipy.sparse import csr_matrix


mp.dps = 500
def generate_correlated_data(n, d, delta, rho):
  M = np.zeros((n, d))
  u = vec(M)
  U = np.identity(n)
  V = np.array([[rho**abs(i-j) for j in range(d)] for i in range(d)])

  M_index = np.array(range(n))
  true_outliers = np.random.choice(M_index, size=5, replace=False)
  M[true_outliers] += delta
  
  
  X = M + stats.matrix_normal.rvs(mean=np.zeros((n,d)), rowcov=U, colcov=V)
  Sigma = np.kron(V,U)
  return X, u, Sigma, true_outliers
def dbscan_inner(is_core, neighborhoods, labels):
    label_num = 0
    stack = []

    for i in range(len(labels)):
        #print(labels[i], label_num, is_core[i])
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # Similar to the classic algorithm for computing connected components,
        # but we label non-core points as part of a cluster (component) without
        # eypanding their neighborhoods.
        while True:
            #change labels for first point => append all posible seeds of that cluster
            #to a stack, change labels until stack empty (no seeds remain)
            #print("change labels")
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    neighb = neighborhoods[i]
                    #print(neighb)
                    for i in range(len(neighb)):
                        #print(v)
                        v = neighb[i]
                        if labels[v] == -1:
                            stack.append(v)
            #print(stack, labels)
            if len(stack) == 0:
                break
            i = stack.pop()
        label_num += 1
    return labels

def euclidean_distance(point1, point2):
   return np.sqrt(np.sum((point1 - point2) ** 2))
#print(y)
def nearest_neighbor(y, eps, minPts):

  neigh = []
  iscore = []
  for i, point1 in enumerate(y):
      distances = []
      for point2 in y:
          dist = np.sqrt(np.sum((point1 - point2) ** 2))
          distances.append(dist)
      #print(distances)
      pos = [i for i in range(len(distances)) if distances[i] <= eps]
      iscore.append(int(len(pos) >= minPts))
      neigh.append(pos)
  return neigh, iscore


def dbscan_sk(y, minPts, eps):
    neighborhoods, is_core = nearest_neighbor(y, eps, minPts)
    #print(neighborhoods, is_core)
    labels = [-1] * len(is_core)  # Initialize labels as noise
    return dbscan_inner(is_core, neighborhoods, labels), neighborhoods

def intersection_of_two_intervals(interval1, interval2):
    # Intersection giữa hai interval đơn lẻ
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    if start < end:
        return (start, end)
    return None  # Không có phần giao

def intersection_of_union_and_interval(union, interval):
    # Intersection giữa union (có thể gồm nhiều interval) và một interval đơn lẻ
    result = []
    for u_interval in union:
        intersect = intersection_of_two_intervals(u_interval, interval)
        if intersect:
            result.append(intersect)
    return result

def find_intersection_of_all_unions(union_list):
    # Bắt đầu với union đầu tiên
    intersection = union_list[0]

    # Lặp qua từng union tiếp theo và tính phần giao
    for union in union_list[1:]:
        new_intersection = []
        for interval in intersection:
            # Tính intersection giữa mỗi interval trong union và interval trong intersection hiện tại
            new_intersection += intersection_of_union_and_interval(union, interval)
        intersection = new_intersection

    return intersection
def pivot_with_specified_interval(z_interval, etaj, etajTy, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        #print(al, etajTy, ar
        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        #print(f"al: {al}, ar: {ar}, tn_mu: {tn_mu}, tn_sigma: {tn_sigma}")
        #print(f"numerator: {round(numerator,10)}, denominator: {round(denominator,10)}")
        #print(f"etaTy: {etajTy}, interval: {z_interval}")
        cdf_al = mp.ncdf((al - tn_mu) / tn_sigma)
        cdf_ar = mp.ncdf((ar - tn_mu) / tn_sigma)

        #print(f"CDF(al): {round(cdf_al,10)}, CDF(ar): {round(cdf_ar,10)}")
    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None


def vec(A):
  vec = A.reshape(-1, order='F')
  return vec.reshape(-1,1)
def unvec(vecA, n, d):
  return vecA.reshape(n, d, order='F')
def solve_quadratic_inequality(a, b, c,seed = 0):
    """ ax^2 + bx +c <= 0 """
    if abs(a) < 1e-8:
        a = 0
    if abs(b) < 1e-8:
        b = 0
    if abs(c) < 1e-8:
        c = 0
    if a == 0:
        # print(f"b: {b}")
        if b > 0:
            # return [(-np.inf, -c / b)]
            return [(-np.inf, np.around(-c / b, 8))]
        elif b == 0:
            # print(f"c: {c}")
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c', seed)
                return 
        else:
            return [(np.around(-c / b, 8), np.inf)]
    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")
    # print("delta:", delta)
    # print(f"2a: {2*a}")
    x1 = (- b - np.sqrt(delta)) / (2*a)
    x2 = (- b + np.sqrt(delta)) / (2*a)
    # if x1 > x2:
    #     x1, x2 = x2, x1  
    x1 = np.around(x1, 8)
    x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]


def interval_intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        
        # Calculate the potential intersection
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        
        # If the interval is valid, add to results
        if start < end:
            result.append((start, end))
        
        # Move the pointer which ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result
def interval_union(a, b):
    # Merge the two sorted interval lists into one sorted list
    merged = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    # Add any remaining intervals from a or b
    merged.extend(a[i:])
    merged.extend(b[j:])
    
    # Merge overlapping intervals
    if not merged:
        return []
    
    result = [merged[0]]
    for current in merged[1:]:
        last = result[-1]
        if current[0] < last[1]:
            # Overlapping or adjacent, merge them
            new_start = last[0]
            new_end = max(last[1], current[1])
            result[-1] = (new_start, new_end)
        else:
            result.append(current)
    return result
  
def compute_z_interval(j_test, n, d, O_obs, eps, neps, a, c, zk, minusO, x_zk):
  #print(len(O), len(minusO))
  I_d = np.identity(d)
  
  trunc_interval = [(-np.inf, np.inf)]

  for j in range(n):
    #euclidean distance
    A1_sparse = []
    A2_sparse = []
    for i in range(n):
      e = np.zeros((1,n))
      e[0][j] = 1
      if i != j:
        e[0][i] = -1
      else:
        continue
      
      Id_x_e = np.kron(I_d, e)
      Id_x_e_sparse = csr_matrix(Id_x_e)
      A_sparse = Id_x_e_sparse.T @ Id_x_e_sparse
      #sparsity = 1.0 - ( np.count_nonzero(A) / float(A.size) )
      #print(sparsity)
      
      if i in neps[j]:
       
        A1_sparse.append(A_sparse)
      else:
        
        A2_sparse.append(A_sparse)
        
    b = eps*eps
    b = np.array(b).reshape(1, 1)
    #print(a.shape, c.shape, z.shape)
    p1 = np.array([a.T @ (Ai @ a) for Ai in A1_sparse]).reshape(1, -1, 1)
    q1 = np.array([c.T @ (Ai @ a) + a.T @ (Ai @ c) for Ai in A1_sparse]).reshape(1,-1,1)
    t1 = np.array([c.T @ (Ai @ c) - b for Ai in A1_sparse]).reshape(1,-1,1)

    #print("p1", p1)
    p2 = -1*(np.array([a.T @ (Ai @ a) for Ai in A2_sparse])).reshape(1, -1, 1)
    q2 = -1*(np.array([c.T @ (Ai @ a) + a.T @ (Ai @ c) for Ai in A2_sparse])).reshape(1,-1,1)
    t2 = -1*(np.array([c.T @ (Ai @ c) - b for Ai in A2_sparse])).reshape(1,-1,1)
    #print("p2", p2)

            
    for k in range(len(A1_sparse)):
      res = solve_quadratic_inequality(p1[0][k][0], q1[0][k][0], t1[0][k][0])
      if res == "No solution":
        print(p1[0][k][0], q1[0][k][0], t1[0][k][0])
      else:
        trunc_interval = interval_intersection(trunc_interval,res)
                
      

    for k in range(len(A2_sparse)):
      res = solve_quadratic_inequality(p2[0][k][0], q2[0][k][0], t2[0][k][0])
      if res == "No solution":
        print(p2[0][k][0], q2[0][k][0], t2[0][k][0])
      else:
        trunc_interval = interval_intersection(trunc_interval,res)


 

  eT_minusO = np.zeros((1, n))
  eT_minusO[:,minusO] = 1

  #print(X)
  #print(x)
  
  eT_mean_minusO = np.kron(I_d, eT_minusO)/len(minusO)
  #print(np.dot(eT_mean_minusO, x), np.mean(X[minusO], axis = 0))

  e_j = np.zeros((1, n))
  e_j[:, j_test] = 1
  temp = np.kron(I_d, e_j) - eT_mean_minusO
  Xj_meanXminusO = np.dot(temp, x_zk)
  #print(Xj_meanXminusO, X[j] - np.mean(X[minusO], axis = 0))

  S = np.sign(Xj_meanXminusO)
  #print(S)
  B = np.multiply(S, temp)
  Ba = np.dot(B, a)
  Bc = np.dot(B, c)
 
  #print(Ba.shape, Bc.shape)
  for j in range (Ba.shape[0]):
    res = solve_quadratic_inequality(0, -Ba[j][0], -Bc[j][0])
    trunc_interval = interval_intersection(trunc_interval,res)
  return trunc_interval, S
  