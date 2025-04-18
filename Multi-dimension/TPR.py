from naive_pvalue import run_naive_TPR
from parametric import run_parametric_TPR
from si_oc import run_sioc_TPR
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm 
from util import *


max_iteration = 1500
max_count = 500
Alpha = 0.05
n = 100
d = 5
minpts = 10
eps = 2
list_delta = [1, 2, 3, 4]


def run_wrapper(args):
    n, d, delta, minpts, eps, method = args
    return method(n, d, delta, minpts, eps)
    
def tpr_si_oc():
    list_TPR = []
    for delta in list_delta:
        number_of_true_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, d, delta, minpts, eps, run_sioc_TPR) for _ in range(max_iteration)]

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
                    if number_of_true_positive + number_of_false_negative == max_count:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        list_TPR.append(TPR)
    return list_TPR
    
def tpr_parametric():
    list_TPR = []
    for delta in list_delta:
        number_of_true_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, d, delta, minpts, eps, run_parametric_TPR) for _ in range(max_iteration)]

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
                    if number_of_true_positive + number_of_false_negative == max_count:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        list_TPR.append(TPR)
    return list_TPR
def tpr_bonferroni():
    list_TPR = []
    p_max = Alpha/(2**n)
    for delta in list_delta:
        number_of_true_positive = 0 
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, d, delta, minpts, eps, run_naive_TPR) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for p_value, true_detection in tqdm(pool.imap_unordered(run_wrapper, args), total=max_iteration):
                if p_value is not None:
                    if p_value <= p_max:
                        if true_detection:
                            number_of_true_positive += 1                    
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        
                    count += 1
                    if number_of_true_positive + number_of_false_negative == max_count:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        list_TPR.append(TPR)
    return list_TPR

if __name__ == '__main__':

    
    list_TPR_SI = tpr_parametric()
    save_list_to_csv(list_TPR_SI, "saved_data/list_TPR_SI.csv")
    list_TPR_bonferroni = tpr_bonferroni()
    save_list_to_csv(list_TPR_bonferroni, "saved_data/list_TPR_bonferroni.csv")
    list_TPR_SI_OC = tpr_si_oc()
    save_list_to_csv(list_TPR_SI_OC, "saved_data/list_TPR_SI_OC.csv")

    fig, ax = plt.subplots()

    ax.plot(list_delta, list_TPR_SI, color = 'green', label = 'SI-CLAD')
    ax.scatter(list_delta, list_TPR_SI, color='green')
    ax.plot(list_delta, list_TPR_SI_OC, color = 'orange', label = 'SI-CLAD-oc')
    ax.scatter(list_delta, list_TPR_SI_OC, color='orange')
    ax.plot(list_delta, list_TPR_bonferroni, color = 'blue', label = 'Bonferroni')
    ax.scatter(list_delta, list_TPR_bonferroni, color='blue')
    
    ax.set_xticks(list_delta)
    ax.set_ylim(-0.05,1.05)
    
    ax.legend(loc = 'upper right')
    plt.xlabel("delta")
    plt.ylabel("TPR")
    
    plt.savefig("Multi-dimension-tpr.png", dpi=300, bbox_inches='tight')
  