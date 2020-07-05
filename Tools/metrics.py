import torch
import numpy as np
import torch.nn.functional as F

class Metric():
    """Base class for all metrics. 
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class PSNR(Metric):
    
    def __init__(self):
        super(PSNR, self).__init__()