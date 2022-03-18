import numpy as np

def detection_rate(labels, predicts):
    """Calculate detection rate (TPR)

    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    """
    # print(f"TP: {np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32)}")
    # print(f"TP + FN: {np.sum(labels == -1)}")
    return np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32) / np.sum(labels == -1)

def false_alarm_rate(labels, predicts):
    """Calculate false alarm rate
    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    """
    # print(f"FP: {np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32)}")
    # print(f"FP + TN: {np.sum(labels == 1)}")
    return np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32) / np.sum(labels == 1)