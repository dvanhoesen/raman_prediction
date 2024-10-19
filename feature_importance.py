import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def return_feature_importances(trained_model, input): 
    # calculate feature importance using layer-wise relevance propagation 