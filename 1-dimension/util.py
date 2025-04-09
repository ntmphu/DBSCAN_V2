import numpy as np
from mpmath import mp

mp.dps = 500
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


#print(y)
def nearest_neighbor(y, eps, minPts):

  y = y.flatten()
  neigh = []
  iscore = []
  for i, point1 in enumerate(y):
      distances = []
      for point2 in y:
          dist = abs(point1 - point2)
          distances.append(dist)

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
def generate(n):
    u = np.zeros((n, 1))
    noise = np.random.normal(loc=0, scale=1, size=n)
    y = u + noise.reshape(-1, 1)
    return y
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

def compute_z_interval(n, O_obs, eps, neps, a, c, zk):
    # Initialize the interval for z
    trunc_interval = [(-np.inf, np.inf)]
    b = eps * eps  # Scalar threshold
    a = a.flatten()
    c = c.flatten()
    # Assume a and c are 1D NumPy arrays of length n
    for j in range(n):
        # Vectorized differences for all i
        diff_a = a[j] - a  # Shape: (n,)
        diff_c = c[j] - c  # Shape: (n,)
        
        for i in range(n):
            if i != j:
                # Compute coefficients directly
                p = diff_a[i] ** 2
                q = 2 * diff_a[i] * diff_c[i]
                t = diff_c[i] ** 2 - b
                
                if i in neps[j]:
                    # Solve p*z² + q*z + t <= 0
                    res = solve_quadratic_inequality(p, q, t)
                else:
                    # Solve -p*z² - q*z - t <= 0 (i.e., p*z² + q*z + t >= 0)
                    res = solve_quadratic_inequality(-p, -q, -t)
                
                if res != "No solution":
                    trunc_interval = interval_intersection(trunc_interval, res)
    
    return trunc_interval


def generate_trueoutliers(n, delta):
    u = np.zeros((n, 1))
    u_index = np.array(range(n))
    true_outliers = np.random.choice(u_index, size=(n//5), replace=False)
    u[true_outliers] += delta
    noise = np.random.normal(loc=0, scale=1, size=n)
    y = u + noise.reshape(-1, 1)
    return y, true_outliers