import numpy as np
from sklearn.metrics import roc_auc_score

FP_COST = 2.0
CHARGEBACK_FEE = 15.0
FRAUD_MULTIPLIER = 1.0

def business_loss(y_true,y_pred_probs,amounts,thresholds):
    y_pred = (y_pred_probs >= thresholds).astype(int)
    fp = np.sum((y_true == 0) & (y_pred==1))
    fn_mask = (y_true == 1) & (y_pred == 0)
    fraud_amount_loss = (amounts[fn_mask] * FRAUD_MULTIPLIER).sum()
    chargeback_loss = fn_mask.sum() * CHARGEBACK_FEE
    total_loss = (fp * FP_COST) + fraud_amount_loss + chargeback_loss

    return float(total_loss)


def evaluate_model(y_true,y_pred_probs,amounts,thresholds):
    auc = roc_auc_score(y_true,y_pred_probs)
    loss = business_loss(y_true,y_pred_probs,amounts,thresholds)

    return {'auc': auc, 'business_loss': loss,'thresholds': thresholds}


def find_best_threshold(y_true,y_probs,amounts):
    thresholds = np.linspace(0.01,0.99,50)

    losses = [
        business_loss(y_true,y_probs,amounts,t)
        for t in thresholds
    ]
    best_idx = int(np.argmin(losses))

    return thresholds[best_idx],losses[best_idx]

