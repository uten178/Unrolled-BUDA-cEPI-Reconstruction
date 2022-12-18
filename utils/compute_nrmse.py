import numpy as np


def compute_nrmse(ground_truth, recon):
    return 100 * np.linalg.norm(ground_truth - recon) / np.linalg.norm(ground_truth)