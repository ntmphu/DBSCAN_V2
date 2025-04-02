from util import *
from parametric import run_parametric
import time
import matplotlib.pyplot as plt
import seaborn as sns

def changing_d():
# Parameters
    n = 100
    d_values = [2,4,6,8]
    minpts = 10
    eps = 1.25


    times = {d: [] for d in d_values}
    len_intervals = {d: [] for d in d_values}
    # Measure computational time
    for i, d in enumerate(d_values):
       
        num_runs = 0
        while num_runs < 10:
            start_time = time.time()
            p, encountered_interval = run_parametric(n=n, d=d, minpts=2*d, eps=eps*(i+1))
            end_time = time.time()
            if p is not None:
                times[d].append(end_time - start_time)
                num_runs += 1
                len_intervals[d].append(encountered_interval)
        

    # Plotting the results using Seaborn
    data_time = [times[d] for d in d_values]
    data_interval = [len_intervals[d] for d in d_values]
    fig, axes  = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(data=data_time, ax = axes[0], width=0.6, color = 'black', medianprops={"color": "r"}, fill = False, showfliers=False)
    
    axes[0].set_xticks(range(len(d_values)))
    axes[0].set_xticklabels(d_values)
    axes[0].set_xlabel('dimension d')
    axes[0].set_ylabel('Computational time(s)')
   
    
    sns.boxplot(data=data_interval, ax = axes[1], width=0.6, color = 'black', medianprops={"color": "r"}, fill = False, showfliers=False)
   
    axes[1].set_xticks(range(len(d_values)))
    axes[1].set_xticklabels(d_values)
    axes[1].set_xlabel('dimension d')
    axes[1].set_ylabel('#intervals')
    
    plt.savefig("time_changing_d_2.png", dpi=300, bbox_inches='tight')
    # Automatically adjust subplot layout to avoid overlap
    
def changing_n():
    # Parameters
    d = 5
    n_values = [50,100,150,200]
    minpts = 10
    eps = 3


    # Dictionary to store computational times for each d
    times = {n: [] for n in n_values}
    len_intervals = {n: [] for n in n_values}
    # Measure computational time
    for n in n_values:
        print(n)
        num_runs = 0
        while num_runs < 10:
            start_time = time.time()
            p, encountered_interval = run_parametric(n=n, d=d, minpts=minpts, eps=eps)
            end_time = time.time()
            if p is not None:
                times[n].append(end_time - start_time)
                num_runs += 1
                len_intervals[n].append(encountered_interval)
        

    # Plotting the results using Seaborn
    data_time = [times[n] for n in n_values]
    data_interval = [len_intervals[n] for n in n_values]
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot: Computational time
    sns.boxplot(data=data_time, ax=axes[0], width=0.6, color = 'black', medianprops={"color": "r"}, fill = False, showfliers=False)  # No outliers
    axes[0].set_xticks(range(len(n_values)))
    axes[0].set_xticklabels(n_values)
    axes[0].set_xlabel('#source instances')
    axes[0].set_ylabel('Computational time (s)')
  

    # Second subplot: Number of intervals
    sns.boxplot(data=data_interval, ax=axes[1],width=0.6, color = 'black', medianprops={"color": "r"}, fill = False, showfliers=False)
    axes[1].set_xticks(range(len(n_values)))
    axes[1].set_xticklabels(n_values)
    axes[1].set_xlabel('#source instances')
    axes[1].set_ylabel('#intervals')
    


    
    plt.savefig("time_changing_n.png", dpi=300, bbox_inches='tight')
    # Automatically adjust subplot layout to avoid overlap

changing_d()
