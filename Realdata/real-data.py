
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
def load_abs():
# Fetch the dataset
    abs = fetch_ucirepo(id=445)
    data = abs.data.original
    data = data.dropna()
    data = data.drop(['ID'], axis='columns')
    return data

def load_heart():
    heart_disease = fetch_ucirepo(id=45)
    data = heart_disease.data.original
    data = data.dropna()
    data = data.drop(['num'], axis='columns')
    return data

def load_breast():
    breast_cancer_data = fetch_ucirepo(id=17)   
    data = breast_cancer_data.data.original
    data = data.dropna()
    data = data.drop(['ID', 'Diagnosis'], axis='columns')
    return data


data = load_abs()
fig_name = "absense"
d = 20
minpts = 2 * d
eps = 6

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

def run_wrapper(args):
    minpts, eps, n, d, func = args
    subset_indices = np.random.choice(data_scaled.shape[0], size=n, replace=False)
    feature_indices = np.random.choice(data_scaled.shape[1], size=d, replace=False)
    data_subset = data_scaled[subset_indices, :][:, feature_indices]
    return func(data_subset, minpts, eps)

if __name__ == '__main__':
    max_iteration = 1000
    max_count = 500
    n = 200
    
    
    list_parametric_pvalue = []
    list_sioc_pvalue = []
    args_parametric  = [(minpts, eps, n, d, run_parametric) for _ in range(max_iteration)]
    args_sioc  = [(minpts, eps, n, d, run_sioc) for _ in range(max_iteration)]

    
    with Pool() as pool:
        # Use tqdm with imap_unordered for real-time progress tracking        
        for selective_p_value in tqdm(pool.imap_unordered(run_wrapper, args_sioc), total=max_iteration):
            if selective_p_value is not None:
                list_sioc_pvalue.append(selective_p_value)
                if len(list_sioc_pvalue) == max_count:
                    break
    save_list_to_csv(list_sioc_pvalue, f"saved_data/{fig_name}_sioc.csv")
    
    with Pool() as pool:
        for selective_p_value in tqdm(pool.imap_unordered(run_wrapper, args_parametric), total=max_iteration):
            if selective_p_value is not None:
                list_parametric_pvalue.append(selective_p_value)
                if len(list_parametric_pvalue)  == max_count:
                    break
    save_list_to_csv(list_parametric_pvalue, f"saved_data/{fig_name}_parametric.csv")
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
    plt.savefig(f'{fig_name}.png', dpi=300, bbox_inches='tight')
    
   

