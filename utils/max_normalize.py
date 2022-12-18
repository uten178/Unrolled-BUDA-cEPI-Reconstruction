import numpy as np


def max_normalize_each_sample_in_batch(x):
    for idx_sample in range(x.shape[0]):
        x[idx_sample] = x[idx_sample]/np.max(np.abs(x[idx_sample]))
    return x
