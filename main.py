import numpy as np
import gym
from naf import NAF
from memory import Memory, Transition
import torch
import os
from utils import stateToTensor
import argparse

parser = argparse.ArgumentParser(description = "RL project")
parser.add_argument('--name', default='run10')
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--batchSize', type=int, default=128)
parser.add_argument('--numEpisodes', type=int, default=100000)
parser.add_argument('--hiddenSize', type=int, default=128)
parser.add_argument('--updatesPerStep', type=int, default=5)
parser.add_argument('--memorySize', type=int, default=2000000)
args = parser.parse_args()

run = args.name
gamma = args.gamma
tau = args.tau
batchSize = args.batchSize
hiddenSize = args.hiddenSize
memorySize = args.memorySize
numEpisodes = args.numEpisodes
updatesPerStep = args.updatesPerStep

if not os.path.exists(f'./models'):
    os.mkdir(f'./models')
    print("Created models")
if not os.path.exists(f'./models/{run}_h{hiddenSize}_b{batchSize}'):
    os.mkdir(f'./models/{run}_h{hiddenSize}_b{batchSize}')
    print(f'Created models/{run}_h{hiddenSize}_b{batchSize}')
print("Folder ready")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('cuda')
else:
    device = torch.device('cpu')
    print('cpu')

memory = Memory(memorySize)
env = gym.make('FetchSlide-v1')
print("Env created")
state = env.reset()
print(f'Env tested: {state}')
agent = NAF(gamma, tau, hiddenSize, env.observation_space["observation"].shape[0] + 3, env.action_space, device)
print("Agent created")
totalNumSteps = 0
updates = 0
rewards = []

for episode in range(numEpisodes):
    state = env.reset()
    print(f'Episode {episode}')
    state = stateToTensor(state).to(device)
    print(f'State to tensor')
    episodeReward = 0
    valueLossEp = 0
    updatesEpisode = 1
    while True:
        print(f'Select action')
        action = agent.selectAction(state, None, None)
        print(f'Selected: {action.cpu().numpy()[0]}')

        nextState, reward, done, _ = env.step(action.cpu().numpy()[0])
        print('Step taken')
        totalNumSteps += 1
        episodeReward += reward

        action =action.cpu().numpy()
        mask = torch.Tensor([not done]).cpu().numpy()
        nextState = stateToTensor(nextState)
        nextStateNumpy = nextState.cpu().numpy()
        reward = torch.Tensor([reward]).cpu().numpy()
        print('Push to memory')
        memory.push(state.cpu().numpy(), action, mask, nextStateNumpy, reward)
        print('Pushed')
        state = nextState.to(device)

        if len(memory) > batchSize:
            print('Training started')
            for _ in range(updatesPerStep):
                transition = memory.sample(batchSize)
                batch = Transition(*zip(*transition))
                updatesEpisode += 1
                valueLoss, policyLoss = agent.updateParameters(batch, device)
                valueLossEp += valueLoss
                updates += 1
        
        if done:
            break
        
    rewards.append(episodeReward)
    checkEvery = 20
    if episode % checkEvery == 0:
        state = env.reset()
        state = stateToTensor(state).to(device=device)
        episodeReward = 0
        while True:
            # env.render()
            action = agent.selectAction(state)
            nextState, reward, done, _ = env.step(action.cpu().numpy()[0])
            episodeReward += reward
            nextState = stateToTensor(nextState).to(device)
            state = nextState
            if done:
                break
        rewards.append(episodeReward)

        agent.saveModel(f"models/{run}_h{hiddenSize}_b{batchSize}/naf_e{episode}.model")
        
        # print(f"Episode: {episode+1}, total numsteps: {totalNumSteps}, reward: {rewards[-1]}, average reward: {np.mean(rewards)}")
        with open(f"models/{run}_h{hiddenSize}_b{batchSize}/{run}_agentTraining.csv", "a+") as f:
            f.write(f'{episode}, {totalNumSteps}, {episodeReward/checkEvery}, {np.mean(rewards)/checkEvery}, {valueLossEp/updatesEpisode}\n')

env.close()
