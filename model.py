import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity
from . import utils

class ModifiedSimCLR(nn.Module):
    def __init__(self, encoder, descriptor_size=256):
        super(ModifiedSimCLR, self).__init__()

        self.encoder = encoder
        self.encoder.fc = Identity()  # Replace the fc layer with an Identity function

        self.detector_head = utils.detector_head()
        self.descriptor_head = utils.descriptor_head(descriptor_size)

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        prob_i = self.detector_head(h_i)
        prob_j = self.detector_head(h_j)
        
        descriptors_i = self.descriptor_head(h_i)
        descriptors_j = self.descriptor_head(h_j)
        
        return prob_i, prob_j, descriptors_i, descriptors_j
