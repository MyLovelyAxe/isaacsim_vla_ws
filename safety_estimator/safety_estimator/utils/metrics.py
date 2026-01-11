"""
This file defines metrics to measure whether a netowrk performs well or not.
"""

import numpy as np

def compute_f1(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
) -> float:
    """Compute F1 score."""

    return 1.0


def compute_accuracy(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
) -> float:
    """Compute accuracy score."""

    return 1.0


def compute_recall(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
) -> float:
    """Compute recall score."""

    return 1.0


def compute_auc(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
) -> float:
    """Compute AUC."""

    return 1.0


def compute_metrics(
    risk_score: np.ndarray,
    risk_label: np.ndarray,
):
    """Compute metrics scores of predicted risk score and ground truth risk label.
    
    :param risk_score: prediction of safety estimator network, shape (batch_size, 1)
    :param risk_label: ground truth label of riskness, shape (batch_size, 1)
    """
    # TODO: whether sigmoid risk_score??
    metrics = dict(
        f1 = compute_f1(risk_score=risk_score, risk_label=risk_label),
        accuracy = compute_accuracy(risk_score=risk_score, risk_label=risk_label),
        recall = compute_recall(risk_score=risk_score, risk_label=risk_label),
        auc = compute_auc(risk_score=risk_score, risk_label=risk_label),
    )
    return metrics



if __name__ == "__main__":

    score = np.random.rand(32,1).astype(np.float32)
    label = np.random.rand(32,1).astype(np.float32)

    metrics = compute_metrics(risk_score=score, risk_label=label)

    print("F1: {f1:5.3f} | Accuracy: {accuracy:5.3f} | "
          "Recall: {recall:5.3f} | AUC: {auc:5.3f} |".format(**metrics),flush=True)

