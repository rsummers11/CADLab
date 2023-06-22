"""
Author : Seung Yeon Shin
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
"""

""" Cross-entropy loss that we use for the ILP loss """

import torch
from torch import nn
    
    
class CrossEntropy(nn.Module):
    """Computes cross entropy
    """

    def __init__(self, epsilon=1e-6):
        super(CrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
        CE = (-target*torch.log(input+self.epsilon) - (1-target)*torch.log(1-input+self.epsilon)).mean()
        
        return CE