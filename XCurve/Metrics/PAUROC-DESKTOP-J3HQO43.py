from sklearn.metrics import roc_auc_score
import numpy as np
import math

def argTopk(score, k):
    return argBottomk(-score, k)

def argBottomk(score, k):
    
    return devec(np.argpartition(devec(score), k-1)[:k])

def vec(x):
    return x.reshape(-1,1)

def devec(x):
    return x.squeeze() if x.shape[0] > 1 else x 

def subIndexNegPosK(score, label, min_tpr, max_fpr):
    negIndex = np.where(label != 1)[0]
    posIndex = np.where(label == 1)[0]
    nP, nN = len(posIndex), len(negIndex)
    kpos = math.floor(nP * min_tpr)
    kneg = math.floor(nN * max_fpr)
    negTopIndex = negIndex[argTopk(score[negIndex], kneg)]
    posBotIndex = posIndex[argBottomk(score[posIndex], kpos)]

    return devec(np.vstack((vec(posBotIndex), vec(negTopIndex))))

def PartialAUROC(y_true, y_pred, min_tpr, max_fpr):
    
    if min_tpr <= 0.0 or min_tpr > 1.0:
        raise ValueError("TPR range alpha mush be ranged (0, 1]")
    if max_fpr <= 0.0 or max_fpr > 1.0:
        raise ValueError("FPR range beta mush be ranged (0, 1]")
    
    subIndex = subIndexNegPosK(y_pred, y_true, min_tpr, max_fpr)
    y_true = y_true[subIndex]
    y_pred = y_pred[subIndex]
    return roc_auc_score(y_true, y_pred)
