import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import datetime, timedelta
import math
import ast
import os

def plot_pie_chart(df):
    job_counts = df['CodeGroup'].astype(str).value_counts()

    top15 = job_counts.head(10)
    others = job_counts.iloc[10:].sum()

    if others > 0:
        job_counts_plot = pd.concat([top15, pd.Series({'Other': others})])
    else:
        job_counts_plot = top15

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(job_counts_plot))]

    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        job_counts_plot,
        labels=[''] * len(job_counts_plot), 
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        colors=colors
    )

    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        ha = 'left' if x > 0 else 'right'
        ax.annotate(
            job_counts_plot.index[i],
            xy=(x, y),
            xytext=(1.2 * x, 1.2 * y),
            ha=ha,
            va='center',
            fontsize=16,
            arrowprops=dict(arrowstyle='-')
        )

    for autotext in autotexts:
        autotext.set_fontsize(16)

    plt.title('Perlmutter GPU Jobs in March 2025: Code Group Distribution', fontsize=18)
    plt.savefig('../plots/code_groups_pie_chart.svg', bbox_inches='tight')
    plt.show()

def plot_job_summary_violin(df):
    summary = {
        'Job Duration (hrs)': df['ElapsedSecs'] / 3600,
        'Wait Time (hrs)': df['WaitTime'] / 3600,
        'Max GPU Utilization (%)': df['gpu_utilization_max'],
        'Max Memory Utilization (%)': df['mem_util_max']
    }

    colors = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00']
    
    n = len(summary)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, n), sharey=False)

    if n == 1:
        axes = [axes]  # ensure axes is always iterable

    for ax, (i, (label, series)) in zip(axes, enumerate(summary.items())):
        clean_data = series.dropna()
        mean_value = clean_data.mean()

        sns.violinplot(y=clean_data, ax=ax, color=colors[i % len(colors)], inner='box')

        ax.set_title(label, fontsize=19)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add mean value as text
        textstr = f"Mean: {mean_value:.2f}"
        ax.text(0.95, 0.95, textstr,
                transform=ax.transAxes,
                fontsize=18,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../plots/summary_violin.svg', bbox_inches='tight')
    plt.show()
    
def plot_util_dist(df, col_name, title, color="#f2b6c6"):
    data = df[col_name].dropna().astype(float).values
    sorted_values = np.sort(data)
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    #PDF
    ax2.hist(data, bins=30, color=color, alpha=0.5, label="PDF", edgecolor="black", hatch='//')
    ax2.set_ylabel("Count", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    #CDF
    ax1.plot(sorted_values, cdf_values * 100, color="black", linewidth=3, label="CDF")
    ax1.axhline(50, color="red", linestyle="--", linewidth=1.5)  # Reference line at 50%
    ax1.set_xlabel(col_name, fontsize=16)
    ax1.set_ylabel("Cumulative Count (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="both", labelsize=14)

    # Percentiles
    percentiles = [10, 25, 75, 95]
    percentile_values = np.percentile(data, percentiles)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for perc, val, col in zip(percentiles, percentile_values, colors):
        ax1.axvline(val, color=col, linestyle='--', linewidth=2, label=f'{perc}th pct: {val:.2f}')

    ax2.legend(loc="upper left", fontsize=12, frameon=True)  # PDF legend on top
    ax1.legend(loc="upper center", fontsize=12, frameon=True)  # CDF legend just below PDF

    ax1.grid(False)
    ax2.grid(False)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../plots/{col_name}_dist.svg', bbox_inches='tight')
    plt.show()

def plot_energy_dist(df, col_name, title, color="#f2b6c6"):
    data = df[col_name].dropna().astype(float).values
    sorted_values = np.sort(data)
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    min_val = max(data.min(), 1e-1)
    max_val = data.max()
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

    # PDF
    ax2.hist(data, bins=bins, color=color, alpha=0.5, label="PDF", edgecolor='black', hatch='o')
    ax2.set_ylabel("Count", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_xscale('log')

    # CDF
    ax1.plot(sorted_values, cdf_values * 100, color="black", linewidth=3, label="CDF")
    ax1.axhline(50, color="red", linestyle="--", linewidth=1.5)  # Reference line at 50%
    ax1.set_xlabel(col_name, fontsize=16)
    ax1.set_ylabel("Cumulative Count (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="both", labelsize=14)

    # Percentiles
    percentiles = [10, 25, 75, 95]
    percentile_values = np.percentile(data, percentiles)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for perc, val, col in zip(percentiles, percentile_values, colors):
        ax1.axvline(val, color=col, linestyle='--', linewidth=2, label=f'{perc}th pct: {val:.2e}')


    ax2.legend(loc="upper left", fontsize=12, frameon=True)  # PDF legend on top
    ax1.legend(loc="upper right", fontsize=12, frameon=True)  # CDF legend just below PDF

    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax2.grid(False)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../plots/{col_name}_dist.svg', bbox_inches='tight')
    plt.show()


def plot_power_dist(df, col_name, title, color="#f2b6c6"):
    data = df[col_name].dropna().astype(float).values
    sorted_values = np.sort(data)
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    min_val = max(data.min(), 1e-1)
    max_val = data.max()
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    
    #PDF
    ax2.hist(data, bins=bins, color=color, alpha=0.5, label="PDF", edgecolor='black', hatch='o')
    ax2.set_ylabel("Count", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    # CDF
    ax1.plot(sorted_values, cdf_values * 100, color="black", linewidth=3, label="CDF")
    ax1.axhline(50, color="red", linestyle="--", linewidth=1.5)  # Reference line at 50%
    ax1.set_xlabel(col_name, fontsize=16)
    ax1.set_ylabel("Cumulative Count (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="both", labelsize=14)

    # Percentiles
    percentiles = [10, 25, 75, 95]
    percentile_values = np.percentile(data, percentiles)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for perc, val, col in zip(percentiles, percentile_values, colors):
        ax1.axvline(val, color=col, linestyle='--', linewidth=2, label=f'{perc}th pct: {val:.2e}')

    ax2.legend(loc="upper left", fontsize=12, frameon=True)  # PDF legend on top
    ax1.legend(loc="upper right", fontsize=12, frameon=True)  # CDF legend just below PDF

    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax2.grid(False)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../plots/{col_name}_dist.svg', bbox_inches='tight')
    plt.show()


def plot_tif_dist(df, col_name, title):
    
    data = df[col_name].values
    sorted_values = np.sort(data)
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    # PDF
    ax2.hist(data, bins=30, color="#332288", alpha=0.5, label="PDF", edgecolor="black")
    ax2.set_ylabel("Count", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    # CDF
    ax1.plot(sorted_values, cdf_values * 100, color="black", linewidth=3, label="CDF")
    ax1.axhline(50, color="red", linestyle="--", linewidth=1.5)  # Reference line at 50%
    ax1.set_xlabel("Temporal Imbalance Factor", fontsize=16)
    ax1.set_ylabel("Cumulative Count (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="both", labelsize=14)

    ax2.legend(loc="upper right", fontsize=14, frameon=True)  # PDF legend on top
    ax1.legend(loc="upper right", fontsize=14, frameon=True, bbox_to_anchor=(1, 0.85))  # CDF legend just below PDF

    ax1.grid(False)
    ax2.grid(False)

    plt.title(title,fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../plots/{col_name}_dist.svg', bbox_inches='tight')
    plt.show()


def plot_sif_dist(df, col_name, title):
    
    data = df[col_name].dropna().values
    sorted_values = np.sort(data)
    #print(sorted_values) #checkpoint
    cdf_values = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    # PDF
    ax2.hist(data, bins=30, color="#88CCEE", alpha=0.5, label="PDF", edgecolor="black")
    ax2.set_ylabel("Count", fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    # CDF
    ax1.plot(sorted_values, cdf_values * 100, color="black", linewidth=3, label="CDF")
    ax1.axhline(50, color="red", linestyle="--", linewidth=1.5)  # Reference line at 50%
    ax1.set_xlabel("Spatial Imbalance Factor", fontsize=16)
    ax1.set_ylabel("Cumulative Count (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="both", labelsize=14)

    ax2.legend(loc="upper right", fontsize=14, frameon=True)  # PDF legend on top
    ax1.legend(loc="upper right", fontsize=14, frameon=True, bbox_to_anchor=(1, 0.85))  # CDF legend just below PDF

    ax1.grid(False)
    ax2.grid(False)

    plt.title(title,fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../plots/{col_name}_dist.svg', bbox_inches='tight')
    plt.show()

def plot_ai_dist(df, col_name, title, label,color = 'purple'):
    data = df[col_name].dropna().values
    data = data[data > 0]  # Remove non-positive values for log-scale

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()

    bin_count = 30
    bin_edges = np.logspace(np.log10(data.min()), np.log10(data.max()), bin_count + 1)
    weights = (np.ones_like(data) / len(data)) * 100

    # Histogram
    ax1.hist(data, bins=bin_edges, color=color, alpha=0.7, edgecolor='none',
             weights=weights, label=label)

    ax1.set_xscale('log')        
    ax1.set_ylim(0.01, 100)
    ax1.set_ylabel('Fraction of Samples (%)', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax1.set_xlabel('AI Performance (TF/s)', fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)

    
    '''
    Reference:
    Austin, B., Kulkarni, D., Cook, B., Williams, S., & Wright, N. J. (2024, November). 
    System-Wide Roofline Profiling-a Case Study on NERSCs Perlmutter Supercomputer. 
    In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, 
    Storage and Analysis (pp. 1398-1404). IEEE.
    '''
    # Roofline Plot
    peak_perf = 78 if col_name == 'AI_fp16' else 19.5 if col_name == 'AI_fp32' else 9.7 if col_name == 'AI_fp64' else 19.5
    dram_bandwidth = 1.555  # GB/s

    ai_range = np.logspace(np.log10(data.min()), np.log10(data.max()), 100)
    roofline_perf = np.minimum(peak_perf, ai_range * dram_bandwidth)

    ax2.plot(ai_range, roofline_perf, 'r-', linewidth=2, label='FP Performance Roofline')
    ax2.set_ylabel('FP Performance (TF/s)', fontsize=14, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')
    
    ai_threshold = peak_perf / dram_bandwidth
    ax1.axvline(ai_threshold, color='black', linestyle='--', linewidth=2, label='Roofline Transition')

    plt.title(title, fontsize=14)
    ax1.grid(True, which='both', alpha=0.3)
    ax2.legend(loc="upper left", fontsize=12, frameon=True)  # PDF legend on top
    ax1.legend(loc="upper left", fontsize=12, frameon=True, bbox_to_anchor=(0, 0.9))  # CDF legend just below PDF


    plt.tight_layout()
    plt.savefig(f'../plots/{col_name}_dist.svg', bbox_inches='tight')
    plt.show()



    