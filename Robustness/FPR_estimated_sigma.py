from parametric import  run_parametric_estimatesigma
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm 
from util import *

max_iteration = 500 #actually 120

list_alpha = [0.05, 0.1]
list_n = [50, 100, 150, 200]
def run_wrapper(args):
    n = args
    return run_parametric_estimatesigma(n)
def fpr_parametric_estimate():
    list_FPR = {A:[] for A in list_alpha}
    for Alpha in list_alpha:
        for n in list_n:
            
            number_of_false_positive = 0
            count = 0
            # Set up multiprocessing Pool and run computations in parallel
            args = [(n) for _ in range(max_iteration)]
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
            list_FPR[Alpha].append(FPR)
    return list_FPR


def plot_fpr(list_FPR, list_n, list_alpha, dist_name):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange']
    labels = [f'alpha = {alpha}' for alpha in list_alpha]

    for i, alpha in enumerate(list_alpha):
        plt.plot(list_n, list_FPR[alpha], color=colors[i], label=labels[i])
        plt.scatter(list_n, list_FPR[alpha], color=colors[i])

    plt.xticks(list_n)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.xlabel("Sample Size (n)")
    plt.ylabel("FPR")
    plt.title(f'FPR ({dist_name})')
    plt.savefig(f'FPR_{dist_name}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  
    list_FPR_estimate = fpr_parametric_estimate()

    # Example of how to call this function with your FPR dictionaries
    plot_fpr(list_FPR_estimate, list_n, list_alpha, 'estimated sigma')
