import torch
import numpy as np

def stateToTensor(state):
    return torch.Tensor([np.concatenate((state["observation"], state["desired_goal"], state["achieved_goal"]))])
