import torch
import numpy as np

def stateToTensor(state):
    return torch.Tensor([np.concatenate((state["observation"], state["achieved_goal"] - state["desired_goal"]))])
