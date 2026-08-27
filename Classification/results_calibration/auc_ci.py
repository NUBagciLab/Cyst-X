import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def calculate_auc_ci(y_true, y_pred_probs, n_bootstraps=1000, ci_level=0.95):
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        y_b, pred_b = resample(y_true, y_pred_probs)
        
        # Check if bootstrap sample has both classes
        if len(np.unique(y_b)) < 2:
            continue
            
        score = roc_auc_score(y_b, pred_b)
        bootstrapped_scores.append(score)
        
    # Calculate 95% CI
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower_bound = np.percentile(sorted_scores, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + ci_level) / 2 * 100)
    
    return lower_bound, upper_bound