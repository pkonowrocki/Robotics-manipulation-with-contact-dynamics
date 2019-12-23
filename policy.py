import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, hiddenSize, numInputs, numOutputs, device):
        super(Policy, self).__init__()
        self.device = device
        self.numOutputs = numOutputs
        self.numInputs = numInputs

        self.V = nn.Sequential(
            nn.BatchNorm1d(numInputs),
            nn.Linear(numInputs, 2*hiddenSize),
            nn.PReLU(2*hiddenSize),
            nn.Linear(2*hiddenSize, hiddenSize),
            nn.PReLU(hiddenSize),
            nn.Linear(hiddenSize, 1))

        self.mu = nn.Sequential(
            nn.BatchNorm1d(numInputs),
            nn.Linear(numInputs, 2*hiddenSize),
            nn.PReLU(2*hiddenSize),
            nn.Linear(2*hiddenSize, hiddenSize),
            nn.PReLU(hiddenSize),
            nn.Linear(hiddenSize, hiddenSize),
            nn.PReLU(hiddenSize),
            nn.Linear(hiddenSize, numOutputs),
            nn.Tanh())

        self.L = nn.Sequential(
            nn.BatchNorm1d(numInputs),
            nn.Linear(numInputs, hiddenSize),
            nn.PReLU(hiddenSize),
            nn.Linear(hiddenSize, numOutputs*numOutputs),
            nn.PReLU(numOutputs*numOutputs),
            nn.Linear(numOutputs*numOutputs, numOutputs*numOutputs))

        self.trilMask = torch.tril(torch.ones(numOutputs, numOutputs), diagonal=-1).unsqueeze(0).to(device)
        self.diagMask = torch.diag(torch.ones(numOutputs, numOutputs)).unsqueeze(0).to(device)


    def forward(self, inputs):
        x, u = inputs

        V = self.V(x)
        mu = self.mu(x)

        Q = None
        if u is not None:
            L = self.L(x).view(-1, self.numOutputs, self.numOutputs).to(self.device)
            L = L * self.trilMask.expand_as(L) + torch.exp(L) * self.diagMask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))
            uMu = (u-mu).unsqueeze(2)
            A = -0.5*torch.bmm(torch.bmm(uMu.transpose(2,1), P), uMu)[:, :, 0]
            Q = A+V

        return mu, Q, V