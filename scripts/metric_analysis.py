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


import sys
sys.path.append('code_group.py')
from code_group import match_name, match_algo


def load_summarize_csv(file_path1, file_path2):
    df1 = pd.read_csv(file_path1, low_memory=False)
    

    column_names = ['JobID', 'Arch', 'User', 'Nodelist', 'Submit', 'Start', 'End', 'exename', 'numcpus', 
                'numnodes', 'numtasks', 'QOS', 'Partition','Account', 'JobName', 'AppName', 'NodeHours'
                ]
    df2 = pd.read_csv(file_path2, delimiter='\t', header=None,names=column_names)
    df2['Submit'] = pd.to_datetime(df['Submit'], unit='s')
    df2['Start'] = pd.to_datetime(df['Start'], unit='s')
    df2['End'] = pd.to_datetime(df['End'], unit='s')
    df2['JobID'] = df2['JobID'].str.split('.').str[0]

    return df


def load_large_query_csv(file_path):
    column_names = ['JobID', 'Arch', 'User', 'Nodelist', 'Submit', 'Start', 'End', 'exename', 'numcpus', 
                    'numnodes', 'numtasks', 'QOS', 'Partition','Account', 'JobName', 'AppName', 'NodeHours'
                    ]

    df = pd.read_csv(file_path, delimiter='\t', header=None,names=column_names)
    df['Submit'] = pd.to_datetime(df['Submit'], unit='s')
    df['Start'] = pd.to_datetime(df['Start'], unit='s')
    df['End'] = pd.to_datetime(df['End'], unit='s')
    df['Wait'] = (df['Start'] - df['Submit']).dt.total_seconds()
    df['ElapsedSecs'] = (df['End'] - df['Start']).dt.total_seconds()
    df = df[~df['JobID'].str.contains(r'\.(ba|ex)')]
    df['CodeGroup'] = df['exename'].apply(match_name)

    return df


def load_summarize_parquet(file_path, time_col='timestamp'):
    df = pd.read_parquet(file_path)

    start_time = pd.to_datetime(df[time_col].iloc[0], unit='ms')
    end_time = pd.to_datetime(df[time_col].iloc[-1], unit='ms')
    print(f"This data covers {start_time} to {end_time}.")

    return df


def find_nodelist(df):
    all_nodes = set()
    for nodelist_str in df['nodes']:
        node_list = ast.literal_eval(nodelist_str)
        all_nodes.update(node_list)
    return list(all_nodes)


def find_dcgm_file_names(df):
    start_time = df['Start'].iloc[0]
    end_time = df['End'].iloc[-1]

    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)

    # Round start time down to nearest 30 minutes
    rounded_start_minutes = (start_time.minute // 30) * 30
    file_start = start_time.replace(minute=rounded_start_minutes, second=0, microsecond=0)

    # Round end time up to nearest 30 minutes
    end_minutes = end_time.minute
    if end_minutes > 0 and end_minutes != 30:
        if end_minutes <= 30:
            rounded_end_minutes = 30
        else:
            rounded_end_minutes = 0
            end_time = end_time + timedelta(hours=1)
    else:
        rounded_end_minutes = end_minutes

    file_end = end_time.replace(minute=rounded_end_minutes, second=0, microsecond=0)

    file_names = []
    current_time = file_start
    while current_time <= file_end:
        next_time = current_time + timedelta(minutes=30)
        start_str = current_time.strftime('%Y-%m-%dT%H:%M:%S-07:00')
        end_str = next_time.strftime('%Y-%m-%dT%H:%M:%S-07:00')
        file_name = f'start_{start_str}_end_{end_str}.pq'
        file_names.append(file_name)
        current_time = next_time

    return file_names


def find_load_dcgm_files(file_names):
    """
    Loads DCGM parquet files from disk if they exist.

    Args:
        file_names (list): List of DCGM parquet file names.

    Returns:
        list: List of loaded DataFrames (one per file found).
    """
    dcgm_df_list = []
    for file_name in file_names:
        file_path = f'/pscratch/sd/b/boztop/set2/{file_name}'
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            dcgm_df_list.append(df)
        else:
            continue
    return dcgm_df_list


def filter_dcgm_files(dcgm_df_list, nodelist):
    filtered_dfs = []
    for df in dcgm_df_list:
        filtered_df = df[df['hostname'].isin(nodelist)]
        if not filtered_df.empty:
            filtered_dfs.append(filtered_df)

    if not filtered_dfs:
        return pd.DataFrame()

    return pd.concat(filtered_dfs, ignore_index=True)




def find_tif_per_job(df, col_name):
    """
    Sencan et al., PEARC'25

    Calculate temporal imbalance factors (TIF) for a given metric per GPU and node.

    This function computes the following statistics for each (gpu_id, hostname) group:
        - Coefficient of Variation (CV): std/mean of the metric.
        - Mean Absolute Change (MAC): mean absolute difference between consecutive values.
        - Trend: absolute value of the linear regression slope.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'gpu_id', 'hostname', and the metric column.
        col_name (str): Name of the column to analyze.

    Returns:
        tuple: (tif_cv, tif_mac, tif_trend)
            tif_cv (float): Maximum coefficient of variation across all groups.
            tif_mac (float): Maximum mean absolute change across all groups.
            tif_trend (float): Maximum trend (slope) across all groups.
            Returns (np.nan, np.nan, np.nan) if insufficient data.
    """
    gpu_node_stats = []
    grouped = df.groupby(['gpu_id', 'hostname'])

    for (gpu, node), group in grouped:
        U = group[col_name].values
        U = U[~np.isnan(U)]
        if len(U) < 2:
            continue

        std_val = np.std(U)
        mean_val = np.mean(U)

        # Linear trend (slope)
        x = np.arange(len(U)).reshape(-1, 1)
        reg = LinearRegression().fit(x, U)
        trend_val = abs(reg.coef_[0])

        # Mean absolute change
        mac_val = np.mean(np.abs(np.diff(U)))

        gpu_node_stats.append({
            'std': std_val,
            'mean': mean_val,
            'trend': trend_val,
            'mac': mac_val
        })

    if not gpu_node_stats:
        return np.nan, np.nan, np.nan

    stats_df = pd.DataFrame(gpu_node_stats)

    max_std = stats_df['std'].max()
    max_mean = stats_df['mean'].max()
    max_trend = stats_df['trend'].max()
    max_mac = stats_df['mac'].max()

    tif_cv = max_std / max_mean if max_mean != 0 else np.nan
    tif_trend = max_trend
    tif_mac = max_mac

    return tif_cv, tif_mac, tif_trend


def find_tif_merged(all_job_data, col_name):
    cvs = all_job_data[f'{col_name}_tif_cv'].values
    macs = all_job_data[f'{col_name}_tif_mac'].values
    trends = all_job_data[f'{col_name}_tif_trend'].values

    df = pd.DataFrame({
        'CV': cvs,
        'MAC': macs,
        'Trend': trends
    })

    original_indices = df.index.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_mask = ~df.isna().any(axis=1)
    if not valid_mask.any():
        return np.full(len(all_job_data), np.nan)

    valid_df = df[valid_mask].copy()
    scaler = MinMaxScaler()
    normalized_valid = scaler.fit_transform(valid_df[['CV', 'MAC', 'Trend']])

    merged_tif = np.full(len(all_job_data), np.nan)
    for orig_idx, (norm_cv, norm_mac, norm_trend) in zip(valid_df.index, normalized_valid):
        merged_tif[orig_idx] = 0.4 * norm_cv + 0.4 * norm_mac + 0.2 * norm_trend

    return merged_tif

def find_sif_per_job(df, col_name):
    # Intra-node Spatial Imbalance Factors
    node_nrs = []
    node_vars = []

    for node, group in df.groupby('hostname'):
        # Mean utilization per GPU within this node
        U = group[col_name].dropna()
        if len(U) < 2:
            continue

        max_val = U.max()
        min_val = U.min()

        if max_val == 0:
            continue
        nr = (max_val - min_val) / max_val
        var = np.var(U)
        node_nrs.append(nr)
        node_vars.append(var)

    intra_nr = max(node_nrs) if node_nrs else np.nan
    intra_var = max(node_vars) if node_vars else np.nan
    intra_merged = (
        (intra_nr + intra_var) / 2
        if not (np.isnan(intra_nr) or np.isnan(intra_var))
        else np.nan
    )

    # Inter-node Spatial Imbalance Factors
    node_means = [
        group[col_name].mean()
        for _, group in df.groupby('hostname')
    ]

    if not node_means:
        return intra_merged, np.nan

    max_val = max(node_means)
    min_val = min(node_means)

    if max_val == 0:
        inter_nr = np.nan
    else:
        inter_nr = (max_val - min_val) / max_val

    inter_var = np.var(node_means)
    inter_merged = (
        (inter_nr + inter_var) / 2
        if not (np.isnan(inter_nr) or np.isnan(inter_var))
        else np.nan
    )

    return intra_merged, inter_merged


def find_sif_normalized(all_job_data, col_name):
    intra = all_job_data[f'{col_name}_sif_intra']
    inter = all_job_data[f'{col_name}_sif_inter']

    df = pd.DataFrame({'intra': intra, 'inter': inter})
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_mask = ~df.isna().any(axis=1)

    normalized_intra_full = np.full(len(df), np.nan)
    normalized_inter_full = np.full(len(df), np.nan)

    valid_df = df[valid_mask]

    if not valid_df.empty:
        intra_scaler = MinMaxScaler()
        normalized_intra = intra_scaler.fit_transform(valid_df[['intra']]).flatten()

        inter_scaler = MinMaxScaler()
        normalized_inter = inter_scaler.fit_transform(valid_df[['inter']]).flatten()

        normalized_intra_full[valid_mask] = normalized_intra
        normalized_inter_full[valid_mask] = normalized_inter

    return normalized_intra_full, normalized_inter_full


def create_combined_df(df, file_name):
    """
    Create a combined DataFrame with job and DCGM metric statistics, and save as a parquet file.

    For each job in the input DataFrame, this function:
        - Loads the corresponding DCGM metrics parquet file (if it exists).
        - Computes temporal and spatial imbalance factors for GPU and memory utilization.
        - Computes summary statistics (min, max, median, mean, std) for selected metrics.
        - Aggregates job info and computed statistics into a single DataFrame.
        - Normalizes and merges imbalance factors.
        - Saves the resulting DataFrame to a parquet file.

    Args:
        df (pd.DataFrame): DataFrame containing job information (must include 'JobID').
        file_name (str): Output file name (without extension) for the combined parquet.

    Returns:
        None. The combined DataFrame is saved to disk.
    """
    job_data_columns = [
        # Job/user info
        'User', 'UID', 'Account', 'Partition', 'Start', 'End', 'JobID', 'NCPUS', 'AllocNodes', 'AllocTRES', 'NodeList',
        'ReservationId', 'ChargeFactor', 'Program', 'Category', 'ElapsedSecs', 'JobName', 'WaitTime', 'TimeLimit',
        'NerscHours', 'RawHours', 'MachineHours', 'QoS',
        # Requested resources
        'req_mem_mb', 'req_gpus', 'req_node', 'req_cpu', 'req_time',
        # DCGM metric summary statistics
        'dram_active_min', 'dram_active_max', 'dram_active_median', 'dram_active_mean', 'dram_active_std',
        'fb_free_min', 'fb_free_max', 'fb_free_median', 'fb_free_mean', 'fb_free_std',
        'fb_used_min', 'fb_used_max', 'fb_used_median', 'fb_used_mean', 'fb_used_std',
        'fp16_active_min', 'fp16_active_max', 'fp16_active_median', 'fp16_active_mean', 'fp16_active_std',
        'fp32_active_min', 'fp32_active_max', 'fp32_active_median', 'fp32_active_mean', 'fp32_active_std',
        'fp64_active_min', 'fp64_active_max', 'fp64_active_median', 'fp64_active_mean', 'fp64_active_std',
        'gpu_utilization_min', 'gpu_utilization_max', 'gpu_utilization_median', 'gpu_utilization_mean', 'gpu_utilization_std',
        'gr_engine_active_min', 'gr_engine_active_max', 'gr_engine_active_median', 'gr_engine_active_mean', 'gr_engine_active_std',
        'nvlink_rx_bytes_min', 'nvlink_rx_bytes_max', 'nvlink_rx_bytes_median', 'nvlink_rx_bytes_mean', 'nvlink_rx_bytes_std',
        'nvlink_tx_bytes_min', 'nvlink_tx_bytes_max', 'nvlink_tx_bytes_median', 'nvlink_tx_bytes_mean', 'nvlink_tx_bytes_std',
        'pcie_rx_bytes_min', 'pcie_rx_bytes_max', 'pcie_rx_bytes_median', 'pcie_rx_bytes_mean', 'pcie_rx_bytes_std',
        'pcie_tx_bytes_min', 'pcie_tx_bytes_max', 'pcie_tx_bytes_median', 'pcie_tx_bytes_mean', 'pcie_tx_bytes_std',
        'sm_active_min', 'sm_active_max', 'sm_active_median', 'sm_active_mean', 'sm_active_std',
        'sm_occupancy_min', 'sm_occupancy_max', 'sm_occupancy_median', 'sm_occupancy_mean', 'sm_occupancy_std',
        'dcgm_tensor_active_min', 'dcgm_tensor_active_max', 'dcgm_tensor_active_median', 'dcgm_tensor_active_mean', 'dcgm_tensor_active_std',
        'dcgm_tensor_hmma_active_min', 'dcgm_tensor_hmma_active_max', 'dcgm_tensor_hmma_active_median', 'dcgm_tensor_hmma_active_mean', 'dcgm_tensor_hmma_active_std',
        'mem_util_min', 'mem_util_max', 'mem_util_median', 'mem_util_mean', 'mem_util_std',
        # Imbalance factors
        'gpu_tif_cv', 'gpu_tif_mac', 'gpu_tif_trend', 'gpu_tif_merged',
        'mem_tif_cv', 'mem_tif_mac', 'mem_tif_trend', 'mem_tif_merged',
        'gpu_sif_intra', 'gpu_sif_inter', 'mem_sif_intra', 'mem_sif_inter',
        # Total utilization metrics
        'total_fp64', 'total_fp32', 'total_fp16', 'total_tensor', 'total_dram_active',
        # AI utilization metrics
        'AI_fp64', 'AI_fp32', 'AI_fp16', 'AI_tensor',
        # Energy metrics
        'avg_power','total_energy_usage_joules', 'total_energy_kwh'
    ]

    job_data = {}

    for idx in range(1, len(df)):
        job_id = df.iloc[idx]['JobID']
        user_name = df.iloc[idx]['User']
        row = df.iloc[idx]
        start_time = row.get('Start')

        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
   
        set1_directory = f'/pscratch/sd/b/boztop/jobwise_dataset/set1/{start_time.month:02d}_{start_time.day:02d}_{start_time.year}'
        pq_file = os.path.join(set1_directory, f"{row.get('JobID')}_{row.get('User')}.pq")

        if os.path.exists(pq_file):
            dcgm_data = pd.read_parquet(pq_file)
        else:
            #print(f"DCGM data file not found for JobID {job_id} in set1. Skipping this job.")
            continue

        set2_directory = f'/pscratch/sd/b/boztop/jobwise_dataset/set2/{start_time.month:02d}_{start_time.day:02d}_{start_time.year}'
        pq_file = os.path.join(set2_directory, f"{row.get('JobID')}_{row.get('User')}.pq")

        if os.path.exists(pq_file):
            dcgm_data_set2 = pd.read_parquet(pq_file)

        else:
            #print(f"DCGM data file not found for JobID {job_id} in set2. Skipping this job.")
            continue
            
        
        # Calculate memory utilization as a percentage
        dcgm_data['nersc_ldms_dcgm_mem_util'] = 100 * (
            dcgm_data['nersc_ldms_dcgm_fb_used'] /
            (dcgm_data['nersc_ldms_dcgm_fb_used'] + dcgm_data['nersc_ldms_dcgm_fb_free'])
        )

        stats = {}

        # Temporal Imbalance Factors
        stats['gpu_tif_cv'], stats['gpu_tif_mac'], stats['gpu_tif_trend'] = find_tif_per_job(
            dcgm_data, 'nersc_ldms_dcgm_gpu_utilization'
        )
        stats['mem_tif_cv'], stats['mem_tif_mac'], stats['mem_tif_trend'] = find_tif_per_job(
            dcgm_data, 'nersc_ldms_dcgm_mem_util'
        )

        # Spatial Imbalance Factors
        stats['gpu_sif_intra'], stats['gpu_sif_inter'] = find_sif_per_job(
            dcgm_data, 'nersc_ldms_dcgm_gpu_utilization'
        )
        stats['mem_sif_intra'], stats['mem_sif_inter'] = find_sif_per_job(
            dcgm_data, 'nersc_ldms_dcgm_mem_util'
        )

        # Summary statistics for selected metrics
        for metric in ['dram_active', 'fb_free', 'fb_used', 'fp16_active', 'fp32_active', 'fp64_active', 'sm_active', 'gpu_utilization', 'gr_engine_active', 'nvlink_rx_bytes', 'nvlink_tx_bytes', 'pcie_rx_bytes', 'pcie_tx_bytes', 'sm_occupancy', 'tensor_active', 'tensor_hmma_active', 'mem_util']:
            col = f"nersc_ldms_dcgm_{metric}"
            stats[f"{metric}_min"] = dcgm_data[col].min()
            stats[f"{metric}_max"] = dcgm_data[col].max()
            stats[f"{metric}_median"] = dcgm_data[col].median()
            stats[f"{metric}_mean"] = dcgm_data[col].mean()
            stats[f"{metric}_std"] = dcgm_data[col].std()


        # Total utilization metrics
        stats['total_fp64'] = dcgm_data['nersc_ldms_dcgm_fp64_active'].sum()
        stats['total_fp32'] = dcgm_data['nersc_ldms_dcgm_fp32_active'].sum()
        stats['total_fp16'] = dcgm_data['nersc_ldms_dcgm_fp16_active'].sum()
        stats['total_tensor'] = dcgm_data['nersc_ldms_dcgm_tensor_active'].sum()
        stats['total_dram_active'] = dcgm_data['nersc_ldms_dcgm_dram_active'].sum()

        # AI utilization metrics
        stats['AI_fp64'] = (9.7 * stats['total_fp64']) / (1.555 * stats['total_dram_active']) if stats['total_dram_active'] > 0 else np.nan
        stats['AI_fp32'] = (19.5 * stats['total_fp32']) / (1.555 * stats['total_dram_active']) if stats['total_dram_active'] > 0 else np.nan
        stats['AI_fp16'] = (78 * stats['total_fp16']) / (1.555 * stats['total_dram_active']) if stats['total_dram_active'] > 0 else np.nan
        stats['AI_tensor'] = (19.5 * stats['total_tensor']) / (1.555 * stats['total_dram_active']) if stats['total_dram_active'] > 0 else np.nan   


        # Extract job info from the input DataFrame
        alloc_tres = row.get('AllocTRES')
        mem_val = 0
        gpu_val = 0
        node_val = 0
        cpu_val = 0
        time_lim = 0
    
        if pd.notna(alloc_tres) and alloc_tres: 
            resources = alloc_tres.split(',')
            for resource in resources:
                if '=' in resource:
                    key, value = resource.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse memory (format: mem=459804M)
                    if key == 'mem':
                        if value.endswith('M'):
                            mem_val = int(value[:-1])  # Remove 'M' and convert to int
                        elif value.endswith('G'):
                            mem_val = int(value[:-1]) * 1024  # Convert GB to MB
                        else:
                            mem_val = int(value)
                    
                    # Parse GPU (format: gres/gpu=8)
                    elif key == 'gres/gpu':
                        gpu_val = int(value)
                    
                    # Parse nodes (format: node=2)
                    elif key == 'node':
                        node_val = int(value)
                    
                    # Parse CPU (format: cpu=256)
                    elif key == 'cpu':
                        cpu_val = int(value)

        job_info = {
            'Start': row.get('Start'),
            'End' : row.get('End'),
            'User': row.get('User'),
            'UID': row.get('UID'),
            'JobID': row.get('JobID'),
            'Account': row.get('Account'),
            'Partition':row.get('Partition'),
            'NCPUS': row.get('NCPUS'),
            'AllocNodes': row.get('AllocNodes'),
            'AllocTRES': row.get('AllocTRES'),
            'NodeList': row.get('NodeList'),
            'ElapsedSecs': row.get('ElapsedSecs'),
            'JobName': row.get('JobName'),
            'WaitTime': row.get('WaitTime'),
            'req_time': row.get('TimeLimit'),
            'NerscHours': row.get('NerscHours'),
            'RawHours': row.get('RawHours'),
            'MachineHours': row.get('MachineHours'),
            'req_mem_mb': mem_val,
            'req_gpus': gpu_val,
            'req_node': node_val,
            'req_cpu': cpu_val,
            'req_time':row.get('TimeLimit'),
            'QoS': row.get('QOS'),
            'ReservationId': row.get('ReservationId'),
            'ChargeFactor': row.get('ChargeFactor'),
            'Program': row.get('program'),
            'Category': row.get('sci_cat')
        }


        # Energy metrics
        # Calculate energy for each interval: Energy (J) = Power (W) * Time (s)
        power_col = 'nersc_ldms_dcgm_power_usage'
        if len(dcgm_data_set2) > 0:
            # Each reading is 10 seconds apart
            energy_joules = (dcgm_data_set2[power_col].sum()) * 10
        else:
            energy_joules = np.nan
        stats['avg_power'] = dcgm_data_set2[power_col].mean() if not dcgm_data_set2.empty else np.nan
        stats['total_energy_usage_joules'] = energy_joules # Total energy in Joules
        stats['total_energy_kwh'] = energy_joules / 3.6e6  # Convert Joules to kWh

        all_data = {**job_info, **stats}
        job_df = pd.DataFrame([all_data], columns=job_data_columns)
        job_df['JobID'] = job_id
        job_data[job_id] = job_df

    if not job_data:
        print("No job data found. No output file created.")
        return

    combined_df = pd.concat(job_data.values(), ignore_index=True)

    # Data cleaning: filter out impossible utilization values
    combined_df = combined_df[combined_df['gpu_utilization_max'] <= 100]
    combined_df = combined_df[combined_df['gpu_utilization_min'] >= 0]
    combined_df = combined_df[combined_df['mem_util_max'] <= 100]
    combined_df = combined_df[combined_df['mem_util_min'] >= 0]

    # Compute merged and normalized imbalance factors
    combined_df['gpu_tif_merged'] = find_tif_merged(combined_df, 'gpu')
    combined_df['mem_tif_merged'] = find_tif_merged(combined_df, 'mem')
    combined_df['gpu_sif_intra_normalized'], combined_df['gpu_sif_inter_normalized'] = find_sif_normalized(combined_df, 'gpu')
    combined_df['mem_sif_intra_normalized'], combined_df['mem_sif_inter_normalized'] = find_sif_normalized(combined_df, 'mem')

    # Save to parquet
    combined_df.to_parquet(f"{file_name}.parquet", index=False)

