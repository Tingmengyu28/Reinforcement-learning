import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import gym                                      # 导入gym
import pygame
import sys


# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
env = gym.make('CartPole-v1', render_mode="human").unwrapped         # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
N_ACTIONS = env.action_space.n                  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]       # 杆子状态个数 (4个)

# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         
        super(Net, self).__init__()                                             

        self.fc1 = nn.Linear(N_STATES, 50)                                     
        self.fc1.weight.data.normal_(0, 0.1)                                   
        self.out = nn.Linear(50, N_ACTIONS)                                     
        self.out.weight.data.normal_(0, 0.1)                                   

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))                                                 
        actions_value = self.out(x)                                             
        return actions_value                                                   


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        # for target updating
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    # Start learning when store transition is full
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                
            self.target_net.load_state_dict(
                self.eval_net.state_dict())         
        self.learn_step_counter += 1                                           

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()                                    
        loss.backward()
        self.optimizer.step()  

dqn = DQN()

for i in range(400):        
    start_learn = False
    print("/r" + '<<<<<<<<<Episode: %s' % i, end = "")
    s = env.reset()[0]         
    episode_reward_sum = 0

    while True:
        env.render()                                                   
        a = dqn.choose_action(s)
        s_, r, done, info, _ = env.step(a)
        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / \
            env.theta_threshold_radians - 0.5
        new_r = r1 + r2
        dqn.store_transition(s, a, new_r, s_)      
        episode_reward_sum += new_r

        s = s_

        if dqn.memory_counter > MEMORY_CAPACITY:              
            print(dqn.memory_counter)
            start_learn = True
            dqn.learn()

        if done:       # 如果done为True
            if start_learn:
                print('/r episode%s---reward_sum: %s' %
                    (i, round(episode_reward_sum, 2)), end = "")
                break
            else:
                break
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
env.close()