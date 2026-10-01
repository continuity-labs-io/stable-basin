import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def run_evaluation_pipeline(X, Y_T, Y_E, repeats=20, n_splits=5, seed=42, n_components=10, shuffle_y=False):
    """
    Evaluates a feature set using a cross-validated Cox Proportional Hazards model with PCA.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Y_T (np.ndarray): Array of survival times or time-to-event for each sample.
        Y_E (np.ndarray): Array of event indicators (1 if event occurred, 0 if censored).
        repeats (int, optional): Number of times to repeat the K-fold cross-validation. Defaults to 20.
        n_splits (int, optional): Number of folds in K-fold cross-validation. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_components (int, optional): Number of principal components to keep from PCA. Defaults to 10.
        shuffle_y (bool, optional): If True, shuffles the outcomes to evaluate random chance (null control). Defaults to False.
        
    Returns:
        tuple[np.ndarray, np.ndarray, int]: A tuple containing:
            - c_indices (np.ndarray): Array of concordance indices for each cross-validation repeat.
            - oof_preds_rep0 (np.ndarray): Out-of-fold predictions from the first repeat.
            - failures (int): Number of failed cross-validation repeats due to convergence or PCA issues.
    """
    X = np.array(X)
    Y_T = np.array(Y_T)
    Y_E = np.array(Y_E)
    
    c_indices = []
    oof_preds_rep0 = np.zeros(len(Y_T))
    failures = 0
    
    for r in range(repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof_preds = np.zeros(len(Y_T))
        
        if shuffle_y:
            np.random.seed(seed + r)
            idx = np.random.permutation(len(Y_T))
            Y_T_curr = Y_T[idx]
            Y_E_curr = Y_E[idx]
        else:
            Y_T_curr = Y_T
            Y_E_curr = Y_E
            
        repeat_failed = False
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_T_train, Y_T_test = Y_T_curr[train_index], Y_T_curr[test_index]
            Y_E_train, Y_E_test = Y_E_curr[train_index], Y_E_curr[test_index]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            n_comp = min(n_components, X_train.shape[1], X_train.shape[0]-1)
            if n_comp < 1:
                repeat_failed = True
                break
                
            pca = PCA(n_components=n_comp, random_state=seed+r)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
            df_train['T'] = Y_T_train
            df_train['E'] = Y_E_train
            
            df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
            
            cph = CoxPHFitter(penalizer=0.1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cph.fit(df_train, duration_col='T', event_col='E')
                preds = cph.predict_partial_hazard(df_test)
                oof_preds[test_index] = preds.values
            except Exception:
                repeat_failed = True
                break
                
        if repeat_failed:
            failures += 1
            continue
            
        if r == 0:
            oof_preds_rep0 = oof_preds.copy()
            
        try:
            c_idx = concordance_index(Y_T_curr, -oof_preds, Y_E_curr)
            c_indices.append(c_idx)
        except Exception:
            c_indices.append(0.5)
        
    return np.array(c_indices), oof_preds_rep0, failures

def run_loco_pipeline(X, Y_T, Y_E, cohorts, seed=42, n_components=10):
    """
    Evaluates a feature set using Leave-One-Cohort-Out (LOCO) cross-validation.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Y_T (np.ndarray): Array of survival times or time-to-event for each sample.
        Y_E (np.ndarray): Array of event indicators (1 if event occurred, 0 if censored).
        cohorts (np.ndarray): Array of cohort labels for each sample to group by.
        seed (int, optional): Random seed for PCA reproducibility. Defaults to 42.
        n_components (int, optional): Number of principal components to keep from PCA. Defaults to 10.
        
    Returns:
        tuple[float, int]: A tuple containing:
            - c_idx (float): The overall concordance index computing using the concatenated out-of-fold predictions.
            - failures (int): Number of failed cohorts due to convergence or PCA issues.
    """
    unique_cohorts = np.unique(cohorts)
    if len(unique_cohorts) < 2:
        return 0.5, 0
    
    oof_preds = np.zeros(len(Y_T))
    oof_preds[:] = np.nan
    failures = 0
    
    for test_cohort in unique_cohorts:
        test_index = np.where(cohorts == test_cohort)[0]
        train_index = np.where(cohorts != test_cohort)[0]
        if len(train_index) == 0:
            continue
            
        X_train, X_test = X[train_index], X[test_index]
        Y_T_train, Y_T_test = Y_T[train_index], Y_T[test_index]
        Y_E_train, Y_E_test = Y_E[train_index], Y_E[test_index]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        n_comp = min(n_components, X_train.shape[1], X_train.shape[0]-1)
        if n_comp < 1:
            failures += 1
            continue
            
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        df_train = pd.DataFrame(X_train_pca, columns=[f"PC{i}" for i in range(X_train_pca.shape[1])])
        df_train['T'] = Y_T_train
        df_train['E'] = Y_E_train
        
        df_test = pd.DataFrame(X_test_pca, columns=[f"PC{i}" for i in range(X_test_pca.shape[1])])
        
        cph = CoxPHFitter(penalizer=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df_train, duration_col='T', event_col='E')
            preds = cph.predict_partial_hazard(df_test)
            oof_preds[test_index] = preds.values
        except Exception:
            failures += 1
            
    valid_idx = ~np.isnan(oof_preds)
    if valid_idx.sum() > 0:
        try:
            c_idx = concordance_index(Y_T[valid_idx], -oof_preds[valid_idx], Y_E[valid_idx])
        except Exception:
            c_idx = 0.5
    else:
        c_idx = 0.5
        
    return c_idx, failures
