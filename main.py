import numpy as np
import gym
from naf import NAF
from memory import Memory, Transition
import torch
import os
from utils import stateToTensor, calcReward
import argparse

desiredGoal = np.array([1.52393724, 0.8054561 , 0.41401894])

parser = argparse.ArgumentParser(description = "RL project")
parser.add_argument('--name', default='run10', help='Name of experiment (default: run10)')
parser.add_argument('--gamma', default=0.99, type=float, help='Gamma value (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, help='Tau value (default: 0.001)')
parser.add_argument('--seed', type=int, default=4, help='')
parser.add_argument('--batchSize', type=int, default=128, help='Learning batch size (default: 128)')
parser.add_argument('--numEpisodes', type=int, default=100000, help='Number of episodes (default: 100000)')
parser.add_argument('--hiddenSize', type=int, default=128, help='Number of hidden neurons (default: 128)')
parser.add_argument('--updatesPerStep', type=int, default=5, help='Number of updates per step (default: 5)')
parser.add_argument('--memorySize', type=int, default=2000000, help='Memory size (default: 2000000)')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose output (default: False)')
parser.add_argument('--checkEvery', type=int, default=20, help='Number of iterations after which model should be tested (default: 20)')
parser.add_argument('--numberOfTests', type=int, default=5, help='Number of tests (default: 5)')
args = parser.parse_args()

run = args.name
gamma = args.gamma
tau = args.tau
batchSize = args.batchSize
hiddenSize = args.hiddenSize
memorySize = args.memorySize
numEpisodes = args.numEpisodes
updatesPerStep = args.updatesPerStep
verbose = args.verbose
checkEvery = args.checkEvery
numberOfTests = args.numberOfTests

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
agent = NAF(gamma, tau, hiddenSize, env.observation_space["observation"].shape[0] + env.observation_space["achieved_goal"].shape[0] + env.observation_space["desired_goal"].shape[0], env.action_space.shape[0] - 1, device)
print("Agent created")
totalNumSteps = 0
updates = 0
rewards = []

for episode in range(numEpisodes):
    shortMemory = Memory(memorySize)
    state = env.reset()
    startingPositionPuck = state["achieved_goal"]
    orginalDistance = np.linalg.norm(startingPositionPuck - desiredGoal)
    state = stateToTensor(state,desiredGoal).to(device)
    valueLossEp = 0
    while True:
        action = agent.selectAction(state, np.random.sample() > 0.2)
        action = action.cpu().numpy()
        nextState, reward, done, _ = env.step(np.concatenate((action[0], [0])))
        totalNumSteps += 1
        state = state.cpu().numpy()
        mask = np.array([not done])
        nextState = stateToTensor(nextState, desiredGoal)
        nextStateNumpy = nextState.cpu().numpy()
        reward = calcReward(state[0,-3:], desiredGoal, orginalDistance)
        episodeReward = reward
        shortMemory.push(state, action, mask, nextStateNumpy, reward) 

        if done:
            break
        else:
            state = nextState.to(device)

        if len(memory) > batchSize:
            for _ in range(updatesPerStep):
                transition = memory.sample(batchSize)
                batch = Transition(*zip(*transition))
                valueLoss = agent.updateParameters(batch, device)
                valueLossEp += valueLoss

    memory.append(shortMemory)
    rewards.append(episodeReward)

    if episode % checkEvery == 0:
        testRewards = []
        for _ in range(numberOfTests):    
            state = env.reset()
            startingPositionPuck = state["achieved_goal"]
            orginalDistance = np.linalg.norm(startingPositionPuck - desiredGoal)
            while True:
                state = stateToTensor(state, desiredGoal).to(device=device)
                #env.render()
                action = agent.selectAction(state, useTarget = True)
                action = action.cpu().numpy()
                nextState, reward, done, _ = env.step(np.concatenate((action[0], [0])))
                episodeReward = calcReward(state.cpu().numpy()[0,-3:], desiredGoal, orginalDistance)
                state = nextState
                if done:
                    break
            rewards.append(episodeReward)
            testRewards.append(episodeReward)

        agent.saveModel(f"models/{run}_h{hiddenSize}_b{batchSize}/naf_e{episode}.model")
        
        if verbose:
            print(f"Episode: {episode}, total numsteps: {totalNumSteps}, test reward: {np.mean(testRewards).item()}, average reward: {np.mean(rewards).item()}, avg (last 5) reward: {np.mean(rewards[:-checkEvery]).item()}")
        with open(f"models/{run}_h{hiddenSize}_b{batchSize}/{run}_agentTraining.csv", "a+") as f:
            f.write(f'{episode}, {totalNumSteps}, {np.mean(testRewards).item()}, {np.mean(rewards).item()}, {np.mean(rewards[:-checkEvery]).item()}, {valueLossEp/(updatesPerStep*batchSize)}\n')

env.close()
