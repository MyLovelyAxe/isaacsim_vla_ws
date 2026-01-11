"""
This file defines metrics to measure whether a netowrk performs well or not.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prepare_inputs(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
    class_thr: float = 0.5,
):
    """ Prepare the prediction and ground truth for computing metrisc.

    :param risk_score: prediction of safety estimator network, shape (batch_size, 1)
    :param risk_label: ground truth label of riskness, shape (batch_size, 1)
    :param class_thr: the threshold for classication of prediction
    """
    logits = np.asarray(risk_score).reshape(-1)
    y_true = np.asarray(risk_label).reshape(-1)

    # Make sure y_true is binary int (0/1)
    y_true = (y_true > 0.5).astype(np.int32)

    # direct output network needs sigmoid to be converted into probability
    # due to loss torch.nn.BCEWithLogitsLoss()
    y_prob = sigmoid(logits)

    # Threshold for class prediction
    y_pred = (y_prob >= class_thr).astype(np.int32)

    return y_true, y_pred, y_prob


def compute_metrics(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
    class_thr: float = 0.5,
):
    """Compute metrics scores of predicted risk score and ground truth risk label.
    
    :param risk_score: prediction of safety estimator network, shape (batch_size, 1)
    :param risk_label: ground truth label of riskness, shape (batch_size, 1)
    :param class_thr: the threshold for classication of prediction
    """

    y_true, y_pred, y_prob = prepare_inputs(risk_score, risk_label, class_thr)

    # If validation set has only one class, roc_auc_score is undefined, if so, return 0.5
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_prob)

    metrics = dict(
        f1 = float(f1_score(y_true, y_pred, pos_label=1)),
        accuracy = float(accuracy_score(y_true, y_pred)),
        recall = float(recall_score(y_true, y_pred, pos_label=1)),
        auc = float(auc),
    )

    return metrics



if __name__ == "__main__":
    # logits
    score = np.random.randn(32, 1).astype(np.float32)
    # binary labels
    label = (np.random.rand(32, 1) > 0.7).astype(np.float32)

    metrics = compute_metrics(score, label)

    print(
        "F1: {f1:5.3f} | Accuracy: {accuracy:5.3f} | "
        "Recall: {recall:5.3f} | AUC: {auc:5.3f} |".format(**metrics),
        flush=True
    )

