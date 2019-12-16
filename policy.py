import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, hiddenSize, numInputs, actionSpace, device):
        super(Policy, self).__init__()
        self.device = device
        self.actionSpace = actionSpace
        numOutputs = actionSpace.shape[0]

        self.batchNorm0 = nn.BatchNorm1d(numInputs)

        self.linear1 = nn.Linear(numInputs, hiddenSize)
        self.batchNorm1 = nn.BatchNorm1d(hiddenSize)

        self.linear2 = nn.Linear(hiddenSize, hiddenSize)
        self.batchNorm2 = nn.BatchNorm1d(hiddenSize)

        self.V = nn.Linear(hiddenSize, 1)

        self.mu = nn.Linear(hiddenSize, numOutputs)

        self.L = nn.Linear(hiddenSize, numOutputs*numOutputs)

        self.trilMask = torch.tril(torch.ones(numOutputs, numOutputs), diagonal=-1).unsqueeze(0).to(device)

        self.diagMask = torch.diag(torch.ones(numOutputs, numOutputs)).unsqueeze(0).to(device)
        self.tanh = torch.tanh

    def forward(self, inputs):
        x, u = inputs
        x = self.batchNorm0(x)
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))

        V = self.V(x)
        mu = self.tanh(self.mu(x))

        Q = None
        if u is not None:
            numOutputs = mu.size(1)
            L = self.L(x).view(-1, numOutputs, numOutputs).to(self.device)
            L = L * self.trilMask.expand_as(L) + torch.exp(L) * self.diagMask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))
            uMu = (u-mu).unsqueeze(2)

            A = -0.5*torch.bmm(torch.bmm(uMu.transpose(2,1), P), uMu)[:, :, 0]

            Q = A+V
        return mu, Q, V