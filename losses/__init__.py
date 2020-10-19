import torch.nn as nn
from .metrics import accuracy
neg_log_llhood = nn.NLLLoss()
