
from sklearn.metrics import roc_curve, auc


def accuracy(preds, targets):
    """Computes the accuracy given a target tensor and a prediction one
    Returns:
       The accuracy
    """

    acc = (preds==targets).sum() / len(preds)
    return acc


def auroc(scores, targets_bin):
    """Computes roc metrics including tpr, fpr, thresholds.
    It also compute the cut-off threshold and the area under the curve
    Args:
        scores: The predictions
        targets_bin: binary targets (known/Unknown)
    Returns:
        auc score, fpr array, tpr aray,\
           threshold array used, best threshold index
    """

    # get stats
    fprs, tprs, thresholds = roc_curve(targets_bin, scores)

    # get cut-off threshold using Youden's J-Score
    cut_off_metric = tprs - fprs
    best_idx = cut_off_metric.argmax()

    # get auroc score
    auroc = auc(fprs, tprs)

    return auroc, fprs, tprs, thresholds, best_idx
