from naive_pvalue import naive_TPR
from parametric import run_parametric_TPR
from si_oc import run_sioc_TPR
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm 
from util import *
import csv
import os

max_iteration = 500
Alpha = 0.05
n = 50
list_delta = [1, 2, 3, 4]

def tpr_si_oc():
    list_TPR = []
    for delta in list_delta:
        number_of_true_positive = 0
        number_of_true_negative = 0
        number_of_false_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, delta) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for naive_p_value, true_detection in tqdm(pool.imap_unordered(run_sioc_TPR, args), total=max_iteration):
                if naive_p_value is not None:
                    if naive_p_value <= Alpha:
                        if true_detection:
                            number_of_true_positive += 1
                        else:
                            number_of_false_positive += 1
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        else:
                            number_of_true_negative += 1
                    count += 1
                    if count == 120:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        list_TPR.append(TPR)
    return list_TPR
    
def tpr_si():
    list_TPR = []
    for delta in list_delta:
        number_of_true_positive = 0
        number_of_true_negative = 0
        number_of_false_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, delta) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for naive_p_value, true_detection in tqdm(pool.imap_unordered(run_parametric_TPR, args), total=max_iteration):
                if naive_p_value is not None:
                    if naive_p_value <= Alpha:
                        if true_detection:
                            number_of_true_positive += 1
                        else:
                            number_of_false_positive += 1
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        else:
                            number_of_true_negative += 1
                    count += 1
                    if count == 120:
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
        number_of_true_negative = 0
        number_of_false_positive = 0
        number_of_false_negative = 0
        count = 0

        # Prepare the arguments as a list of tuples
        args = [(n, delta) for _ in range(max_iteration)]

        # Use multiprocessing to handle (n, delta) pairs
        with Pool() as pool:
            for naive_p_value, true_detection in tqdm(pool.imap_unordered(naive_TPR, args), total=max_iteration):
                if naive_p_value is not None:
                    if naive_p_value <= p_max:
                        if true_detection:
                            number_of_true_positive += 1
                        else:
                            number_of_false_positive += 1
                    else:
                        if true_detection:
                            number_of_false_negative += 1
                        else:
                            number_of_true_negative += 1
                    count += 1
                    if count == 120:
                        break

        # Calculate the True Positive Rate (TPR)
        TPR = number_of_true_positive / (number_of_true_positive + number_of_false_negative)
        list_TPR.append(TPR)
    return list_TPR
def save_list_to_csv(data, filename):
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
        print(f"Saved {filename} successfully.")
    except Exception as e:
        print(f"Error saving {filename}: {str(e)}")

if __name__ == '__main__':
  
    
    list_TPR_SI = tpr_si()
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
    
    # Add scatter points on each line
    
    
    
    

    ax.set_xticks(list_delta)
    ax.set_ylim(-0.05,1.05)
    
    ax.legend(loc = 'upper right')
    plt.xlabel("delta")
    plt.ylabel("TPR")
    
    plt.savefig("1-dimension-TPR.png", dpi=300, bbox_inches='tight')