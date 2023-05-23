import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

class A2C_policy(nn.Module):
    '''
    Policy neural network
    '''
    def __init__(self, input_shape, n_actions):
        super(A2C_policy, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU())

        self.mean_l = nn.Linear(32, n_actions)
        self.mean_l.weight.data.mul_(0.1)

        self.var_l = nn.Linear(32, n_actions)
        self.var_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        # x = x.unsqueeze(0)
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))
    
class A2C_value(nn.Module):
    '''
    Actor neural network
    '''
    def __init__(self, input_shape):
        super(A2C_value, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))


    def forward(self, x):
        return self.lp(x.float())


def test_game(tst_env, agent_policy, test_episodes):
    '''
    Execute test episodes on the test environment
    '''

    reward_games = []
    steps_games = []
    for _ in range(test_episodes):
        obs = tst_env.reset()[0]
        # print(obs)
        rewards = 0
        steps = 0
        while True:
            # ag_mean = agent_policy(torch.tensor(obs))
            ag_mean = agent_policy(torch.tensor(obs, dtype=torch.float32))

            action = np.clip(ag_mean.data.cpu().numpy().squeeze(), -1, 1)
            # print(action)

            next_obs, reward, done, *info = tst_env.step(action)
            steps += 1
            obs = next_obs
            rewards += reward
            # print(done)

            if done:
                reward_games.append(rewards)
                steps_games.append(steps)
                obs = tst_env.reset()[0]
                break

    return np.mean(reward_games), np.mean(steps_games)

ENV_NAME = 'BipedalWalker-v3'
TRAJECTORY_SIZE = 2049
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2

POLICY_LR = 0.0004
VALUE_LR = 0.001


device = 'cpu'


## Test Hyperparameters
test_episodes = 1
save_video_test = True
# N_ITER_TEST = 100

load_model = True
checkpoint_name = "checkpoints/checkpoint_PPO_BipedalWalker-v3_23_4.50.22_0.0004_0.001_2049_64.pth.tar"

if __name__ == '__main__':

    # create the test environment
    test_env = gym.make(ENV_NAME,render_mode='human')
    # if save_video_test:
    #     test_env = gym.wrappers.Monitor(test_env,  "VIDEOS/TEST_VIDEOS_"+writer_name, video_callable=lambda episode_id: episode_id%10==0)

    # initialize the actor-critic NN
    # print(test_env.observation_space.shape, test_env.action_space.shape)
    agent_policy = A2C_policy(test_env.observation_space.shape, test_env.action_space.shape[0]).to(device)
    agent_value = A2C_value(test_env.observation_space.shape).to(device)

    # initialize policy and value optimizer
    optimizer_policy = optim.Adam(agent_policy.parameters(), lr=POLICY_LR)
    optimizer_value = optim.Adam(agent_value.parameters(), lr=VALUE_LR)

     # Do you want to load a trained model?
    if load_model:
        print('> Loading checkpoint {}'.format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        agent_policy.load_state_dict(checkpoint['agent_policy'])
        agent_value.load_state_dict(checkpoint['agent_value'])
        optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        optimizer_value.load_state_dict(checkpoint['optimizer_value'])

    # if n_iter % N_ITER_TEST == 0:
            # print('yes')
        test_rews, test_stps = test_game(test_env, agent_policy, test_episodes)
            # print('yes')
            # print(' > Testing..', n_iter,test_rews, test_stps)
            # if it achieve the best results so far, save the models
            # if test_rews > best_test_result:
            #     torch.save({
            #         'agent_policy': agent_policy.state_dict(),
            #         'agent_value': agent_value.state_dict(),
            #         'optimizer_policy': optimizer_policy.state_dict(),
            #         'optimizer_value': optimizer_value.state_dict(),
            #         'test_reward': test_rews
            #     }, 'checkpoints/checkpoint_'+writer_name+'.pth.tar')
            #     best_test_result = test_rews
            #     print('=> Best test!! Reward:{:.2f}  Steps:{}'.format(test_rews, test_stps))