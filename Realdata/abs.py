
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import sklearn
import numpy as np
from util import *
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
from parametric import run_parametric
from si_oc import run_sioc

# Fetch the dataset
heart_disease = fetch_ucirepo(id=445)

# data (as pandas dataframes)
data = heart_disease.data.original
data = data.dropna()
data = data.drop(['ID'], axis='columns')
# Extract features and set up indices

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Define the wrapper function
def run_wrapper(args):
    minpts, eps, n, func = args
    # Randomly select a subset of data points and features
    subset_indices = np.random.choice(data_scaled.shape[0], size=n, replace=False)
    data_subset = data_scaled[subset_indices]

    return func(data_subset, minpts, eps)
if __name__ == '__main__':
    max_iteration = 200
    n = 200
    minpts = 2 * data_scaled.shape[1]
    eps = 6
    
    list_parametric_pvalue = []
    list_sioc_pvalue = []
    args_parametric  = [(minpts, eps, n, run_parametric) for _ in range(max_iteration)]
    args_sioc  = [(minpts, eps, n, run_sioc) for _ in range(max_iteration)]

    filepathpara = r'C:\Users\phung\OneDrive\Desktop\Statistic Machine Learning\Parametric DBSCAN\Realdata\parametric.txt'
    filepathsioc = r'C:\Users\phung\OneDrive\Desktop\Statistic Machine Learning\Parametric DBSCAN\Realdata\sioc.txt'
    with open(filepathpara, 'a') as file:
        file.write(f'=======\nabsence, minpts = {minpts}, eps = {eps}, n = {n}\n')
    # Set up multiprocessing Pool and run computations in parallel
    with open(filepathsioc, 'a') as file:
        file.write(f'=======\nabsence, minpts = {minpts}, eps = {eps}, n = {n}\n')
    
    with Pool() as pool:
        # Use tqdm with imap_unordered for real-time progress tracking        
        for selective_p_value in tqdm(pool.imap_unordered(run_wrapper, args_sioc), total=max_iteration):
            if selective_p_value is not None:
                list_sioc_pvalue.append(selective_p_value)
                if len(list_sioc_pvalue) == 120:
                    break
    with Pool() as pool:
        for selective_p_value in tqdm(pool.imap_unordered(run_wrapper, args_parametric), total=max_iteration):
            if selective_p_value is not None:
                list_parametric_pvalue.append(selective_p_value)
                if len(list_parametric_pvalue)  == 120:
                    break
    # Boxplot with direct plt commands
    fig, ax = plt.subplots()

# Create the boxplot
    box = ax.boxplot(
        [list_sioc_pvalue, list_parametric_pvalue], 
        tick_labels=['SI-CLAD-oc', 'SI-CLAD'], 
        patch_artist=True,  # Enable box color fill
        boxprops=dict(linewidth=1.5),  # Box border width
        medianprops=dict(color='orange', linewidth=1.5),
        showfliers=False# Median line customization
    )

    # Set custom colors for the boxes
    colors = ['lightskyblue', 'peachpuff']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Title and axis labels

    ax.set_ylabel('p-value')
    # Set the y-axis limits (range 0 - 1)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()

    # Display the plot
    plt.savefig("abs.png", dpi=300, bbox_inches='tight')
    
   

