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
            
            p, encountered_interval, runtime = run_parametric(n=n, d=d, minpts=2*d, eps=eps*(i+1))
            if p is not None:
                times[d].append(runtime)
                num_runs += 1
                len_intervals[d].append(encountered_interval)
        

    # Plotting the results using Seaborn
    save_list_to_csv(times, "time_changing_d.csv")
    save_list_to_csv(len_intervals, "inteval_changing_d.csv")
    
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
    
    plt.savefig("time_changing_d.png", dpi=300, bbox_inches='tight')
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
            
            p, encountered_interval, runtime = run_parametric(n=n, d=d, minpts=minpts, eps=eps)
           
            if p is not None:
                times[n].append(runtime)
                num_runs += 1
                len_intervals[n].append(encountered_interval)
                
    save_list_to_csv(times, "time_changing_n.csv")
    save_list_to_csv(len_intervals, "inteval_changing_n.csv")

    # Plotting the results using Seaborn
    data_time = [times[n] for n in n_values]
    data_interval = [len_intervals[n] for n in n_values]
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # First subplot: Computational time
    sns.boxplot(data=data_time, ax=axes[0], width=0.6, color = 'black', medianprops={"color": "r"}, fill = False, showfliers=False)  # No outliers
    axes[0].set_xticks(range(len(n_values)))
    axes[0].set_xticklabels(n_values)
    axes[0].set_xlabel('Sample size (n)')
    axes[0].set_ylabel('Computational time (s)')
  

    # Second subplot: Number of intervals
    sns.boxplot(data=data_interval, ax=axes[1],width=0.6, color = 'black', medianprops={"color": "r"}, fill = False, showfliers=False)
    axes[1].set_xticks(range(len(n_values)))
    axes[1].set_xticklabels(n_values)
    axes[1].set_xlabel('Sample size (n)')
    axes[1].set_ylabel('#intervals')
    


    
    plt.savefig("time_changing_n.png", dpi=300, bbox_inches='tight')
    # Automatically adjust subplot layout to avoid overlap

changing_d()
changing_n()
