
import os
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision


def ensure_dir(dir : str):
    """Creates a directory

    Args:
        dir (str): the path of the directory
    """
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_experiment(opt, seed : int, experiment_dir : str):
    """Setup seeds, experiment folder
    Args:
        opt : experiment settings
        seed (int): seed number
        experiment_dir (str):path to experiment folder
        batch_size (int): size of the batch
    """
    # Setup SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

    random.seed(seed)
    torchvision.torch.manual_seed(seed)
    torchvision.torch.cuda.manual_seed(seed)

    # Create directory
    ensure_dir(experiment_dir)
