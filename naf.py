import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import policy as p

def softUpdate(target, source, tau):
    for targetParam, sourceParam in zip(target.parameters(), source.parameters()):
        targetParam.data.copy_(targetParam.data * (1.0 - tau) + sourceParam.data*tau)

def hardUpdate(target, source):
    softUpdate(target, source, 1.0)

class NAF:
    def __init__(self, gamma, tau, hiddenSize, numInputs, actionSpace, device = torch.device('cpu')):
        self.actionSpace = actionSpace
        self.numInputs = numInputs
        self.model = p.Policy(hiddenSize, numInputs, actionSpace).to(device=device)
        self.target = p.Policy(hiddenSize, numInputs, actionSpace).to(device=device)
        self.optimizer = Adam(self.model.parameters())
        self.gamma = gamma
        self.tau = tau

        hardUpdate(self.target, self.model)

    def selectAction(self, state, actionNoise = None, paramNoise = None):
        self.model.eval()
        mu, _, _ = self.model((Variable(state), None))
        self.model.train()
        mu = mu.data
        if actionNoise is not None:
            mu += torch.Tensor(actionNoise.noise())

        return mu.clamp(-1, 1)

    def updateParameters(self, batch, device):
        stateBatch = torch.cat(torch.Tensor(batch.state)).to(device)
        actionBatch = torch.cat(torch.Tensor(batch.action)).to(device)
        rewardBatch = torch.cat(torch.Tensor(batch.reward)).to(device)
        maskBatch = torch.cat(torch.Tensor(batch.mask)).to(device)
        nextStateBatch = torch.cat(torch.Tensor(batch.nextState)).to(device)

        _, _, nextStateValues = self.target((nextStateBatch, None))

        rewardBatch = rewardBatch.unsqueeze(1)
        maskBatch = maskBatch.unsqueeze(1)

        expectedStateActionValues = rewardBatch + (self.gamma * maskBatch + nextStateValues)

        _, stateActionValues, _ = self.model((stateBatch, actionBatch))

        loss = F.mse_loss(stateActionValues, expectedStateActionValues)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        softUpdate(self.target, self.model, self.tau)

        return loss.item(), 0

    def saveModel(self, modelPath):
        torch.save(self.model.state_dict(), modelPath)
        
    def loadModel(self, modelPath):
        self.model.load_state_dict(torch.load(modelPath))
