import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import policy as p
import numpy as np

def softUpdate(target, source, tau):
    for targetParam, sourceParam in zip(target.parameters(), source.parameters()):
        targetParam.data.copy_(targetParam.data * (1.0 - tau) + sourceParam.data*tau)

def hardUpdate(target, source):
    softUpdate(target, source, 1.0)

class NAF:
    def __init__(self, gamma, tau, hiddenSize, numInputs, actionSpace, device = torch.device('cpu')):
        self.device = device
        self.actionSpace = actionSpace
        self.numInputs = numInputs
        self.model = p.Policy(hiddenSize, numInputs, actionSpace, device).to(device=device)
        self.target = p.Policy(hiddenSize, numInputs, actionSpace, device).to(device=device)
        hardUpdate(self.target, self.model)
        self.gamma = gamma
        self.tau = tau
        self.optimizer = Adam(self.model.parameters())
        self.loss = torch.nn.MSELoss(reduction='sum')

    def selectAction(self, state, actionNoise = False):
        self.model.eval()
        mu, _, _ = self.model((state, None))
        self.model.train()
        mu = mu.data
        if actionNoise:
            mu += torch.Tensor(np.random.standard_normal(mu.shape)).to(self.device)
        return mu.clamp(-1, 1)

    def updateParameters(self, batch, device):
        #Sample a random minibatch of m transitions
        stateBatch = torch.Tensor(np.concatenate(batch.state)).to(device)
        actionBatch = torch.Tensor(np.concatenate(batch.action)).to(device)
        rewardBatch = torch.Tensor(np.concatenate(batch.reward)).to(device)
        maskBatch = torch.Tensor(np.concatenate(batch.mask)).to(device)
        nextStateBatch = torch.Tensor(np.concatenate(batch.nextState)).to(device)
        
        #Set y_i = r_i + gamma*V'(x_t+1 | Q')
        _, _, nextStateValues = self.target((nextStateBatch, None))
        rewardBatch = rewardBatch.unsqueeze(1)
        maskBatch = maskBatch.unsqueeze(1)
        expectedStateActionValues = rewardBatch + (self.gamma * maskBatch + nextStateValues)

        #Update Q by minimizing the loss
        _, stateActionValues, _ = self.model((stateBatch, actionBatch))
        loss = self.loss(stateActionValues, expectedStateActionValues)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        #Update the target network Q'
        softUpdate(self.target, self.model, self.tau)
        return loss.item()

    def saveModel(self, modelPath):
        torch.save(self.model.state_dict(), modelPath)
        
    def loadModel(self, modelPath):
        self.model.load_state_dict(torch.load(modelPath))
