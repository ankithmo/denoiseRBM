
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import os.path as osp
from os import mkdir

# Driver module
def main(storage=osp.join("..", "results"), MI_storage=osp.join("..", "MI_results"), device=True):
    """
        denoiseRBM

        Arguments:
            - storage: Absolute path to the results of denoiseRBM
            - MI_storage: Absolute path to the results of denoiseRBM-MI
            - device (bool): Indicate whether or not CUDA is available

    """
    # Create storage directories
    if not osp.exists(storage):
        mkdir(storage)
    if not osp.exists(MI_storage):
        mkdir(MI_storage)

    # Device
    device = torch.device(f"cuda{args.device}" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()