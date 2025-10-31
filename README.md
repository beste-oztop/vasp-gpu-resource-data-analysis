# Resource Utilization Analysis and Prediction of VASP GPU Jobs Running on Perlmutter
This is the repository for Resource Utilization Analysis on GPU Jobs Running on Perlmutter, with a specific focus on VASP (Vienna Ab initio Simulation Package) jobs from March 2025.

## Overview
This analysis contains various scripts and functions to analyze GPU resource utilization, as well as power consumption of VASP GPU jobs. The analysis includes plotting functions, data processing, and performance metrics. We provide insights into AI performance metrics and roofline analysis. We also provide a prediction framework for maximum GPU and GPU memory utilization, and the average power consumption of VASP GPU jobs using their job submission parameters.

## Dataset Characteristics
We limit our analysis to VASP jobs that ran on regular and premimum GPU nodes on Perlmutter from March 2025. Using our analysis and prediction methods, one can extend our findings to other GPU applications and workloads.

Our dataset includes:
- Accounting job data for GPU jobs from Slurm, and,
- GPU utilization metrics from NVIDIA Data Center Management Tool (DCGM).

The total number of VASP jobs included in this analysis is **32322**.
Resource utilization statistics overview:
- Memory range: 229902 - 34485300 MB
- GPU range: 4 - 600 GPUs
- Node range: 1 - 150 nodes
- CPU range: 128 - 19200 CPUs
- Avg power range: 50.97953709198813 - 35184372088959.11 W

## Organization
The repository is organized into the following directories:
- `notebooks/`: Contains the following Jupyter notebooks:
  - Resource Utilization Analysis
    - Includes GPU and GPU memory utilization analysis, power consumption analysis, AI Performance Metrics and Roofline Analysis
  - Prediction Framework for GPU Utilization and Power Consumption
    - Includes data preprocessing, model training, evaluation, and prediction functions for **before job execution** resource prediction. 
- `scripts/`: Contains the Python script for **during job execution** power prediction using real-time GPU utilization metrics.
- `plots/`: Contains generated plots from the analysis



