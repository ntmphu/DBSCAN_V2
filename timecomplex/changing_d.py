from multiprocessing import Pool
from tqdm import tqdm
from util import *
from parametric import run_parametric
import time
import matplotlib.pyplot as plt
import seaborn as sns

def run_parametric_wrapper_d(args):
    """
    Unpack arguments and call your real run_parametric.
    Returns (p, encountered_interval, runtime).
    """
    n, d, minpts, eps = args
    return run_parametric(n=n, d=d, minpts=minpts, eps=eps)

if __name__ == "__main__":
    n = 100
    d_values = [2, 4, 6, 8]
    eps_values = [0.6, 2, 2.75, 3.75]
    runs_per_d = 100       # target successful runs per d

    times = {d: [] for d in d_values}
    len_intervals = {d: [] for d in d_values}

    for i, d in enumerate(d_values):
        # Compute the per‑d parameters
        minpts_d = 2 * d
        eps_d = eps_values[i]

        # Over‑allocate: launch twice as many tasks as needed
        task_args = [(n, d, minpts_d, eps_d)] * (runs_per_d * 10)

        with Pool() as pool:
            for p_val, interval, runtime in tqdm(
                    pool.imap_unordered(run_parametric_wrapper_d, task_args),
                    total=len(task_args),
                    desc=f"Parametric runs for d={d}"
                ):
                if p_val is not None:
                    times[d].append(runtime)
                    len_intervals[d].append(interval)
                    # Stop once we have enough good runs
                    if len(times[d]) >= runs_per_d:
                        break

    # After this, `times` and `len_intervals` are populated exactly as before,
    # but all the runs for each d were done in parallel.
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