import gc
import pdb
import sys

import logging
import torch
from tqdm import tqdm


def clean():
    gc.collect()
    torch.cuda.empty_cache()
