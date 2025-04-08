from naive_pvalue import run_naive
from parametric import run_parametric
from si_oc import run_sioc
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm 
from util import *

max_iteration = 200
Alpha = 0.05
n = 100
d = 10
minpts = 10
eps = 3
list_rho = [0.2, 0.4, 0.6, 0.8]
num_cores = multiprocessing.cpu_count()
num_workers = num_cores

def run_wrapper(args):
        n, d, minpts, eps, rho, method = args
        return method(n, d, minpts, eps, rho)
    
def fpr_parametric():
    
    list_FPR = []
    for rho in list_rho:
        args = [(n, d, minpts, eps, rho, run_parametric) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0
        # Set up multiprocessing Pool and run computations in parallel
        with Pool(processes=num_workers) as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        number_of_false_positive += 1
                    count += 1
                    if count == 120:
                        break

        # Calculate the False Positive Rate (FPR)c
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR
def fpr_naive():
    list_FPR = []
    for rho in list_rho:
        args = [(n, d, minpts, eps, rho, run_naive) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0

        # Set up multiprocessing Pool and run computations in parallel
        with Pool(processes=num_workers) as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        number_of_false_positive += 1
                    count += 1
                    if count == 120:
                        break

        # Calculate the False Positive Rate (FPR)
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR
def fpr_bonferroni():
    list_FPR = []
    for rho in list_rho:
        args = [(n, d, minpts, eps, rho, run_naive) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0
        p_max = Alpha/(2**n)

        # Set up multiprocessing Pool and run computations in parallel
        with Pool(processes=num_workers) as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= p_max:
                        number_of_false_positive += 1
                    count += 1
                    if count == 120:
                        break

        # Calculate the False Positive Rate (FPR)
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR
def fpr_si_oc():
    list_FPR = []
    for rho in list_rho:
        args = [(n, d, minpts, eps, rho, run_sioc) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0

        # Set up multiprocessing Pool and run computations in parallel
        with Pool(processes=num_workers) as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        number_of_false_positive += 1
                    count += 1
                    if count == 120:
                        break

        # Calculate the False Positive Rate (FPR)
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR


if __name__ == '__main__':
    list_FPR_SI = fpr_parametric()
    save_list_to_csv(list_FPR_SI, 'list_FPRchangingrho_SI.csv')

    list_FPR_naive = fpr_naive()
    save_list_to_csv(list_FPR_naive, 'list_FPRchangingrho_SI_naive.csv')

    list_FPR_bonferroni = fpr_bonferroni()
    save_list_to_csv(list_FPR_bonferroni, 'list_FPRchangingrho_SI_bonferroni.csv')

    list_FPR_SI_OC = fpr_si_oc()
    save_list_to_csv(list_FPR_SI_OC, 'list_FPRchangingrho_SI_SI_OC.csv')
    
    fig, ax = plt.subplots()
    

    fig, ax = plt.subplots()
    ax.plot(list_rho, list_FPR_SI, color='green', label='SI-CLAD')
    ax.scatter(list_rho, list_FPR_SI, color='green')
    ax.plot(list_rho, list_FPR_SI_OC, color='orange', label='SI-CLAD-oc')
    ax.scatter(list_rho, list_FPR_SI_OC, color='orange')
    ax.plot(list_rho, list_FPR_bonferroni, color='blue', label='Bonferroni')
    ax.scatter(list_rho, list_FPR_bonferroni, color='blue')
    ax.plot(list_rho, list_FPR_naive, color='red', label='Naive')
    ax.scatter(list_rho, list_FPR_naive, color='red')
    
    list_FPR_no_inference = [1,1,1,1]
    ax.plot(list_rho, list_FPR_no_inference, color='purple', label='No-Inference')
    ax.scatter(list_rho, list_FPR_no_inference, color='purple')

    
    ax.set_xticks(list_rho)
    ax.set_ylim(-0.05,1.05)
    
    ax.legend(loc = 'upper right')
  
    plt.xlabel("Correlation coefficient (rho)")
    plt.ylabel("FPR")    
    
    # Save the figure
    plt.savefig("FPR-changing-rho.png", dpi=300, bbox_inches='tight')
 