import torch
from torch import pi
import visualization
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy

l = torch.tensor([[1.0,2],[3,4]])
a = torch.sin(torch.pi*2/3)
print(type(a))
print([e for e in l])
print(l[0,0])