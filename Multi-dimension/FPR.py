from naive_pvalue import run_naive
from parametric import run_parametric
from si_oc import run_sioc
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm 
from util import *

max_iteration = 120
Alpha = 0.05
list_n = [50, 100, 150, 200]
d = 10
minpts = 10
eps = 3

def run_wrapper(args):
        n, d, minpts, eps, method = args
        return method(n, d, minpts, eps)
    
def fpr_parametric():
    
    list_FPR = []
    for n in list_n:
        args = [(n, d, minpts, eps, run_parametric) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0
        # Set up multiprocessing Pool and run computations in parallel
        with Pool() as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        number_of_false_positive += 1
                    count += 1

        # Calculate the False Positive Rate (FPR)c
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR
def fpr_naive():
    list_FPR = []
    for n in list_n:
        args = [(n, d, minpts, eps, run_naive) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0

        # Set up multiprocessing Pool and run computations in parallel
        with Pool() as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= Alpha:
                        number_of_false_positive += 1
                    count += 1

        # Calculate the False Positive Rate (FPR)
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR
def fpr_bonferroni():
    list_FPR = []
    for n in list_n:
        args = [(n, d, minpts, eps,  run_naive) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0
        p_max = Alpha/(2**n)

        # Set up multiprocessing Pool and run computations in parallel
        with Pool() as pool:
            # Use tqdm with imap_unordered for real-time progress tracking
            for p_value in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= p_max:
                        number_of_false_positive += 1
                    count += 1

        # Calculate the False Positive Rate (FPR)
        FPR = number_of_false_positive / count
        list_FPR.append(FPR)
    return list_FPR
def fpr_si_oc():
    list_FPR = []
    for n in list_n:
        args = [(n, d, minpts, eps, run_sioc) for _ in range(max_iteration)]
        number_of_false_positive = 0
        count = 0

        # Set up multiprocessing Pool and run computations in parallel
        with Pool() as pool:
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
  
    
    
    list_FPR_naive = fpr_naive()
    list_FPR_bonferroni = fpr_bonferroni()
    list_FPR_SI_OC = fpr_si_oc()
    list_FPR_SI = fpr_parametric()
    
    fig, ax = plt.subplots()
    

    ax.plot(list_n, list_FPR_SI, color = 'green', label = 'SI-CLAD')
    ax.plot(list_n, list_FPR_SI_OC, color = 'orange', label = 'SI-CLAD-oc')
    ax.plot(list_n, list_FPR_bonferroni, color = 'blue', label = 'Bonferroni')
    ax.plot(list_n, list_FPR_naive, color = 'red', label = 'Naive')
    list_FPR_no_inference = [1,1,1,1]
    ax.plot(list_n, list_FPR_no_inference, color = 'purple', label = 'No-Inference')
    ax.scatter(list_n, list_FPR_no_inference, color = 'purple')
    

    ax.scatter(list_n, list_FPR_SI, color = 'green')
    ax.scatter(list_n, list_FPR_naive, color = 'red')
    ax.scatter(list_n, list_FPR_bonferroni, color = 'blue')
    ax.scatter(list_n, list_FPR_SI_OC, color = 'orange')
    
    ax.set_xticks(list_n)
    ax.set_ylim(-0.05,1.05)
    
    ax.legend(loc = 'upper right')
    plt.xlabel('Sample size (n)')
    plt.ylabel('FPR')
    
    plt.savefig("Multi-dimension-fpr.png", dpi=300, bbox_inches='tight')
  