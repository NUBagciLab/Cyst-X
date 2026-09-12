import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def calculate_auc_ci(y_true, y_pred_probs, n_bootstraps=1000, ci_level=0.95):
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        y_b, pred_b = resample(y_true, y_pred_probs)
        
        # Check if bootstrap sample has both classes
        if len(np.unique(y_b)) < len(np.unique(y_true)):
            continue
        # For 2-class or multi-class macro
        if len(np.unique(y_true)) > 2:
            auc = roc_auc_score(y_b, pred_b, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_b, pred_b)   
        bootstrapped_scores.append(auc)
        
    # Calculate 95% CI
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_bound = np.percentile(sorted_scores, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + ci_level) / 2 * 100)
    
    return lower_bound, upper_bound

def calculate_auc_ci_cv(folds_data, n_bootstraps=1000, ci_level=0.95):
    """
    folds_data: list of tuples [(y_true_fold1, y_prob_fold1), (y_true_fold2, y_prob_fold2), ...]
    """
    bootstrapped_means = []
    k_folds = len(folds_data)
    
    for _ in range(n_bootstraps):
        fold_aucs = []
        valid_sample = True
        
        for y_true, y_prob in folds_data:
            y_b, pred_b = resample(y_true, y_prob)
            if len(np.unique(y_b)) < len(np.unique(y_true)):
                valid_sample = False
                break
            
            # For 2-class or multi-class macro
            if len(np.unique(y_true)) > 2:
                auc = roc_auc_score(y_b, pred_b, multi_class="ovr", average="macro")
            else:
                auc = roc_auc_score(y_b, pred_b)
            fold_aucs.append(auc)
            
        if valid_sample and len(fold_aucs) == k_folds:
            bootstrapped_means.append(np.mean(fold_aucs))
            
    sorted_means = np.sort(bootstrapped_means)
    lower_bound = np.percentile(sorted_means, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(sorted_means, (1 + ci_level) / 2 * 100)
    
    return lower_bound, upper_bound