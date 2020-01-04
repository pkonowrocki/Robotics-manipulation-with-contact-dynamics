import torch
import numpy as np

def stateToTensor(state, goal):
    return torch.Tensor([np.concatenate((state["observation"], goal, state["achieved_goal"]))])
