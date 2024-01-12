# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


import torch
import torch.nn as nn
from torchvision.datasets import Cityscapes

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------





dataset = Cityscapes('./data/', split='train', mode='fine',
                      target_type='semantic')



fig,ax=plt.subplots(ncols=2,figsize=(12,8))
ax[0].imshow(dataset[200][0])
ax[1].imshow(dataset[200][1],cmap='gray')

