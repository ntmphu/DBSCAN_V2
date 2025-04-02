from naive_pvalue import run_naive_TPR
from parametric import run_parametric_TPR
from si_oc import run_sioc_TPR
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm 
from util import *
from mpmath import mp

# Set the desired precision
mp.dps = 500  # 


max_iteration = 500
Alpha = 0.05
n = 100
list_d = [2,4,6,8]
minpts = 10
eps = {2:1.5, 4:2.5, 6:3, 8:3.5}
delta = 1

filepath = r'C:\Users\phung\OneDrive\Desktop\Statistic Machine Learning\Parametric DBSCAN\Multi-dimension\tprchangingd.txt'
        


def run_wrapper(args):
    n, d, delta, minpts, eps, method = args
    return method(n, d, delta, minpts, eps)
    
def tpr_si_oc():
    list_TPR = []
    with open(filepath, 'a') as file:
            file.write(f'sioc\n')
    for i, d in enumerate(list_d):
        number_of_true_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, d, delta, minpts, eps[d], run_sioc_TPR) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for p_value, true_detection in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        if true_detection:
                            number_of_true_positive += 1
                    
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        
                    count += 1
                    if count == 120:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        with open(filepath, 'a') as file:
            file.write(f'd = {d}, TPR = {TPR}, count = {count}\n')
        list_TPR.append(TPR)
    return list_TPR
    
def tpr_parametric():
    list_TPR = []
    with open(filepath, 'a') as file:
            file.write(f'parametric\n')
    for i, d in enumerate(list_d):
        number_of_true_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, d, delta, minpts, eps[d], run_parametric_TPR) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for p_value, true_detection in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        if true_detection:
                            number_of_true_positive += 1
                    
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        
                    count += 1
                    if count == 120:
                        break
        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        with open(filepath, 'a') as file:
            file.write(f'd = {d}, TPR = {TPR}, count = {count}\n')
        list_TPR.append(TPR)
    return list_TPR
def tpr_bonferroni():
    with open(filepath, 'a') as file:
            file.write(f'bonferroni\n')

    list_TPR = []
    p_max = mp.mpf(Alpha) / (2 ** n) 
    for i, d in enumerate(list_d):
        number_of_true_positive = 0 
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, d, delta, minpts, eps[d], run_sioc_TPR) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for p_value, true_detection in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    p_value = mp.mpf(p_value)
                    if p_value <= p_max:
                        if true_detection:
                            number_of_true_positive += 1
                    
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        
                    count += 1
                    if count == 120:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        with open(filepath, 'a') as file:
            file.write(f'd = {d}, TPR = {TPR}, count = {count}\n')
        list_TPR.append(TPR)
    return list_TPR

if __name__ == '__main__':

    
    list_TPR_bonferroni = tpr_bonferroni()
    list_TPR_SI_OC = tpr_si_oc()
    list_TPR_SI = tpr_parametric()

    fig, ax = plt.subplots()

    ax.plot(list_d, list_TPR_SI, color = 'green', label = 'SI-CLAD')
    ax.plot(list_d, list_TPR_SI_OC, color = 'orange', label = 'SI-CLAD-oc')
    ax.plot(list_d, list_TPR_bonferroni, color = 'blue', label = 'Bonferroni')
    
    # Add scatter points on each line
    ax.scatter(list_d, list_TPR_SI, color='green')
    ax.scatter(list_d, list_TPR_SI_OC, color='orange')
    ax.scatter(list_d, list_TPR_bonferroni, color='blue')
    
    ax.set_xticks(list_d)
    ax.set_ylim(-0.05,1.05)
    ax.legend(loc = 'upper right')
  
    plt.xlabel("Dimension (d)")
    plt.ylabel("TPR")    
    
    # Save the figure
    plt.savefig("TPR-changing-d.png", dpi=300, bbox_inches='tight')
    