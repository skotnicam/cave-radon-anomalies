import os
import random

import numpy as np
import torch


def setup_pipeline(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    return args.device
