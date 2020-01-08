import torch
import numpy as np

def stateToTensor(state, goal):
    return torch.Tensor([np.concatenate((state["observation"], goal, state["achieved_goal"]))])

def calcReward(state, goal, orginalDistance):
    currentDistance = np.linalg.norm(goal - state)
    return np.array([-np.expm1(currentDistance/orginalDistance)])
