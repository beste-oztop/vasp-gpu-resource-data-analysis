from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import OrdinalEncoder

import xgboost as xgb
from xgboost.callback import EarlyStopping
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

import optuna

import argparse

from metric_analysis import check_metrics, clean_and_compare

import wandb
from wandb.integration.xgboost import WandbCallback
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV




def prepare_multivariate_sequences(metrics_df, dcgm_input_features,
                                   target_metric='nersc_ldms_dcgm_power_usage',
                                   window=3, #30 seconds of prediction
                                   ):
    # print(metrics_df.head())
    X_list = []
    y_list = []

    for _, row in metrics_df.iterrows():
        s1 = row.get('Set1Metrics')
        s2 = row.get('Set2Metrics')
        #print(s2[target_metric])
        
        arrs = []
        lengths = []

        for f in dcgm_input_features:
            feat = f'nersc_ldms_dcgm_{f}'
            #print(feat)
            arr = np.asarray(s1.get(feat, []), dtype=float)
            if arr.size == 0: 
                continue
            arrs.append(arr)
            lengths.append(len(arr))
        
        # Add previous avg power metric into prediction too!
        arr = np.asarray(s2.get('nersc_ldms_dcgm_power_usage', []), dtype=float)
        if arr.size == 0: 
            continue
        arrs.append(arr)
        lengths.append(len(arr))    
            
        target_arr = np.asarray(s2.get(target_metric,[]), dtype=float)
        lengths.append(len(target_arr))

        n_timesteps = min(lengths)
        usable = n_timesteps - window
        
        if usable <= 0:
            continue
        
        stacked = np.vstack([a[:n_timesteps] for a in arrs]).T  # shape (n_timesteps, n_features_total)

        static_vals = []
        for s_feat in ['User', 'JobName', 'Account', 'Category', 'req_node','req_time']:
            val = row.get(s_feat)
            static_vals.append(float(val) if val is not None else 0.0)
        static_vals = np.array(static_vals, dtype=float)

        # --- build time series sequences
        for i in range(usable):
            seq_window = stacked[i:i+window]  # (window, n_features)
            seq_flat = seq_window.flatten()   # (window * n_features)
            full_features = np.concatenate([seq_flat, static_vals])  # append static features
            X_list.append(full_features)
            y_list.append(target_arr[i+window])

    X = np.stack(X_list)   # (samples, window, n_features)
    y = np.array(y_list).reshape(-1, 1)

    # print(f"Input data shape: X = {X.shape}, y = {y.shape}")
    return X, y

def bin_power(y):
    return pd.cut(
        y.ravel(),
        bins=[-np.inf, 100, 150, 200, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
def train_xgb_classifier(X_train, y_train, X_test, y_test, dcgm_set1_metrics, model_type="XGB", wandb_run_name="XGB_classifier_run"):
    # We construct this as a classification problem
    y_train_cls = bin_power(y_train)
    y_test_cls  = bin_power(y_test)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    X_train_flat = np.nan_to_num(X_train_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_flat  = np.nan_to_num(X_test_flat, nan=0.0, posinf=0.0, neginf=0.0)

    valid_train_idx = ~np.isnan(y_train_cls)
    valid_test_idx  = ~np.isnan(y_test_cls)
    
    X_train_flat = X_train_flat[valid_train_idx]
    y_train_cls  = y_train_cls[valid_train_idx]
    
    X_test_flat  = X_test_flat[valid_test_idx]
    y_test_cls   = y_test_cls[valid_test_idx]


    print("Your are in train_xgb_classifier funtion!!")
    print(f"Cleaned input data shape: X_train = {X_train_flat.shape}, y = {y_train_cls.shape}")

    # --- WandB run for tuning ---
    wandb.init(
        project="runtime_power_pred",
        name=wandb_run_name,
        reinit=True  # allows re-running multiple times in same process
    )

    config_params = {
    "objective": "multi:softprob",
    "num_class": 4,
    "use_label_encoder": False,
    "tree_method": "hist",      # on CPU
    "random_state": 42,
    "verbosity": 0,
    "n_estimators": 300,       # previously 300
    "learning_rate": 0.01,
    "max_depth": 9              # previously 5
    }

    wandb.config.update(config_params)
    xgb_clf = XGBClassifier(**config_params)
 
    xgb_clf.fit(
        X_train_flat, y_train_cls.ravel(),
        eval_set=[(X_train_flat, y_train_cls), (X_test_flat, y_test_cls)],
        eval_metric='mlogloss',
        early_stopping_rounds=100,
        callbacks=[WandbCallback()]
    )

    y_pred_xgb = xgb_clf.predict(X_test_flat)

    acc = accuracy_score(y_test_cls, y_pred_xgb)
    cm = confusion_matrix(y_test_cls, y_pred_xgb)
    print("\n***** XGBoost Classifier Results *****")
    print("Accuracy:", acc)
    print(cm)
    print(classification_report(y_test_cls, y_pred_xgb))

    # --- Log to wandb ---
    wandb.log({
        "accuracy": acc,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test_cls,
            preds=y_pred_xgb,
            class_names=[0, 1, 2, 3]
        )
    })

    importances = xgb_clf.feature_importances_

    # build flattened feature names to match X_train_flat ordering (time-major then feature)
    n_timesteps = X_train.shape[1]
    dcgm_set1_metrics.append('nersc_ldms_dcgm_power_usage')  # previous power metric included
    feature_names = []
    for t in range(n_timesteps):
        for feat in dcgm_set1_metrics:
            feature_names.append(f"{feat}_t{t}")    
    print(f"Total feature names: {len(feature_names)}")

    # Log to wandb
    # after fit and after you built feature_names
    score = xgb_clf.get_booster().get_score(importance_type='gain')  # or 'weight', 'cover'

    # build an importance array aligned with feature_names (default 0 for missing features)
    n_feats = len(feature_names)
    imp_array = np.zeros(n_feats, dtype=float)
    for k, v in score.items():  # keys like 'f12'
        try:
            idx = int(k[1:])
        except ValueError:
            continue
        if 0 <= idx < n_feats:
            imp_array[idx] = float(v)

    # normalize importances to sum to 1 (if total > 0)
    total = imp_array.sum()
    if total > 0:
        norm_imp = imp_array / total
    else:
        norm_imp = imp_array  # all zeros

    # map back to feature names and sort
    mapped = {feature_names[i]: norm_imp[i] for i in range(n_feats)}
    sorted_feats = sorted(mapped.items(), key=lambda x: x[1], reverse=True)

    # print top features
    print("Top features (normalized gain):")
    for feat, val in sorted_feats[:20]:
        print(f"{feat}: {val:.4f}")

    # log as a wandb table
    fi_df = pd.DataFrame(sorted_feats, columns=['feature', 'importance'])
    wandb.log({"feature_importance_table": wandb.Table(dataframe=fi_df)})
    wandb.finish()

    
    model_path = f"../ml_training_files/{model_type}_model.pkl"  
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_clf, f)

    model_dict = {
        'model': xgb_clf,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_xgb,
        'confusion_matrix': cm,
        'model_path': model_path
    }

    dict_path = f"../ml_training_files/{model_type}_results.pkl"
    with open(dict_path, 'wb') as f:
        pickle.dump(model_dict, f)

    cm = confusion_matrix(y_test_cls, y_pred_xgb, labels=[0,1,2,3], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm.astype(int),
                                  display_labels=[0, 1, 2, 3])
    cmap = make_single_color_cmap('#AA4499', 'cmap3')
    disp.plot(cmap=cmap, values_format='d')
    plt.title("XGBoost Classification Confusion Matrix")
    cm_path = f"{model_type}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    return model_dict 
    
if __name__ == "__main__":
    # Get the code group to train from CLI
    parser = argparse.ArgumentParser(description="Script for model training")
    parser.add_argument("CodeGroup", help="Code Group (e.g. VASP, LAMMPS..)")
    parser.add_argument("Model", help="Time series prediction model (e.g. XGB, LSTM..)")
    parser.add_argument("RunName", help="Name of the experiment")
    parser.add_argument("--num_jobs", type=int, help="Number of jobs to train (optional)")
    args = parser.parse_args()

    # Read DCGM data
    chunks = [pd.read_parquet(f"../ml_training_files/{args.CodeGroup}_dcgm_data_part_{i+1}.parquet") for i in range(10)]
    app_job_df = pd.concat(chunks, ignore_index=True)
    app_job_df["MetricsCheck"] = app_job_df.apply(check_metrics, axis=1)
    ok_df = app_job_df[app_job_df["MetricsCheck"].apply(lambda x: x.get("status") == "ok")].copy()    
    
    print(f"Combined DataFrame shape: {ok_df.shape}")

    if args.num_jobs:
        exp_df = ok_df.iloc[-args.num_jobs:]
    else:
        exp_df = ok_df.copy()

    cleaned_df = exp_df.apply(clean_and_compare, axis=1)

    dcgm_set1_metrics =['dram_active', 
                    'fb_free', 'fb_used',
                    'fp64_active', 'tensor_active', 
                    'gpu_utilization', 'mem_util',
                    'sm_active','sm_occupancy',
                    'AI_fp64', 'AI_tensor'
                    ]
    slurm_features = ['User','Category', 'JobName', 'Account', 'req_node','req_time']
    
    print(f"Using DCGM Set1 metrics: {dcgm_set1_metrics}, Length: {len(dcgm_set1_metrics)}")

    train_df, test_df = train_test_split(cleaned_df,
                                        test_size=0.2,   
                                        random_state=42,
                                        shuffle=False
                                        )

    cat_cols = ['User', 'Category', 'JobName', 'Account']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    X_train, y_train = prepare_multivariate_sequences(
        train_df, dcgm_set1_metrics,
        target_metric='nersc_ldms_dcgm_power_usage',
        window=3
    )
    
    X_test, y_test = prepare_multivariate_sequences(
        test_df, dcgm_set1_metrics, 
        target_metric='nersc_ldms_dcgm_power_usage',
        window=3
    )

    if args.Model=='XGB':
        train_features = slurm_features + dcgm_set1_metrics
        model_dict = train_xgb_classifier(X_train, y_train, X_test, y_test, train_features, "XGB", f"{args.RunName}")    
    else : 
        print("Choose XGB as model name!")
        # We can implement other ML methods using the same pipeline

        
    
    