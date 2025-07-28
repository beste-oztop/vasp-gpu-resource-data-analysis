# vasp-gpu-resource-data-analysis
This is the repository for Resource Utilization Analysis on GPU Jobs Running on Perlmutter, with a specific focus on VASP (Vienna Ab initio Simulation Package) jobs from March 2025.

## Overview
This analysis contains various scripts and functions to analyze GPU resource utilization, focusing on VASP jobs. The analysis includes plotting functions, data processing, and performance metrics. We also provide insights into AI performance metrics and roofline analysis. 

## Dataset Characteristics
We limit our analysis to VASP jobs that ran on regular and premimum GPU nodes on Perlmutter from MArch 2025.

The dataset includes:
- GPU utilization metrics from NVIDIA Data Center Management Tool (DCGM)
- Accounting job data for GPU jobs from Slurm

Total number of VASP jobs included in this analysis: 32322
Utilization statistics:
- Memory range: 229902 - 34485300 MB
- GPU range: 4 - 600 GPUs
- Node range: 1 - 150 nodes
- CPU range: 128 - 19200 CPUs
- Avg power range: 50.97953709198813 - 35184372088959.11 W

## Organization
The repository is organized into the following directories:
- `dcgm_slurm_studies/`: Contains scripts for plotting and analyzing GPU utilization data
- `plots/`: Contains generated plots from the analysis
- `notebooks/`: Contains Jupyter notebooks for interactive analysis and visualization



