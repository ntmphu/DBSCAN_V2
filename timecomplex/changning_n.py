from multiprocessing import Pool
from tqdm import tqdm
from util import *
from parametric import run_parametric
import time
import matplotlib.pyplot as plt
import seaborn as sns

def run_parametric_wrapper(args):
    """
    Unpack arguments and call your real run_parametric.
    Returns (p, encountered_interval, runtime).
    """
    n, d, minpts, eps = args
    return run_parametric(n=n, d=d, minpts=minpts, eps=eps)

if __name__ == "__main__":
    d = 5
    n_values = [50, 100, 150, 200]
    minpts = 10
    eps = 3
    runs_per_n = 100

    # Prepare storage
    times = {n: [] for n in n_values}
    len_intervals = {n: [] for n in n_values}

    for n in n_values:
        # we don’t know how many will return p is not None,
        # so we over‑allocate twice as many tasks to be safe
        task_args = [(n, d, minpts, eps)] * (runs_per_n * 5)

        with Pool() as pool:
            for p_val, interval, runtime in tqdm(
                    pool.imap_unordered(run_parametric_wrapper, task_args),
                    total=len(task_args),
                    desc=f"Parametric runs for n={n}"
                ):
                if p_val is not None:
                    times[n].append(runtime)
                    len_intervals[n].append(interval)
                    # stop once we have enough successful runs
                    if len(times[n]) >= runs_per_n:
                        break

    # at this point, `times` and `len_intervals` are filled just like before,
    # but you got all 10 runs per n in parallel!
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