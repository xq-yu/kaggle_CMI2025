import numpy as np
from sklearn.metrics import f1_score

def compute_metric(y_true, y_pred):

    target_gestures = {0,1,2,3,4,5,6,7}
    binary_true = np.array([1 if y in target_gestures else 0 for y in y_true])
    binary_pred = np.array([1 if pred in target_gestures else 0 for pred in y_pred])

    binary_f1 = f1_score(binary_true, binary_pred,pos_label=True,zero_division=0,average='binary')

    macro_true = np.where(np.isin(y_true, list(target_gestures)), y_true, len(target_gestures))
    macro_pred = np.where(np.isin(y_pred, list(target_gestures)), y_pred, len(target_gestures))
    macro_f1 = f1_score(macro_true, macro_pred, average='macro',zero_division=0)
    
    return (binary_f1 + macro_f1) / 2,binary_f1 , macro_f1  # 最终得分