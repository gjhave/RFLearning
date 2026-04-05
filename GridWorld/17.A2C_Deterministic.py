"""
Initialization: 
    A given behavior policy β(a|s). 
    A deterministic target policy μ(s, θ0) where θ0 is the initial parameter. 
    A value function q(s, a, w0) where w0 is the initial parameter. αw, αθ > 0. 
Goal: 
    Learn an optimal policy to maximize J(θ).  
    
At time step t in each episode, do 
    Generate at following β and then observe rt+1, st+1. 
    TD error:  
        δt = rt+1 + γq(st+1, μ(st+1, θt), wt) − q(st, at, wt) 
    Actor (policy update):  
        θt+1 = θt + αθ∇θμ(st, θt)(∇aq(st, a, wt))|a=μ(st) 
    Critic (value update):  
        wt+1 = wt + αwδt∇wq(st, at, wt)
"""

from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import torch
import torch.nn as nn


class A2CDetministic(GridWorldEnv):
    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.device ="mps" if torch.backends.mps.is_available() else "cpu"
        self.target_policy_network = nn.Sequential(
            nn.Linear(2, 64),  # 输入状态（x, y）
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.action_space)),
            nn.Softmax(dim=-1),  # 输出动作概率分布
        ).to(self.device)
        self.function_weights = np.random.random(
            len(["x", "y", "x*y", "x^2", "y^2", "a", "a^2","1"])
        )  # 线性函数逼近的权重，包含x,y,1三个特征的权重
        self.lr = 0.0001
        self.episode_length = 1000
        self.policy_optimizer = torch.optim.Adam(self.target_policy_network.parameters(), lr=self.lr)    



    def q_feature(self, norm_x, norm_y, norm_a):
        return np.array(
            [
                norm_x,
                norm_y,
                norm_x * norm_y,
                norm_x**2,
                norm_y**2,
                norm_a, 
                norm_a**2,
                1,
            ]
        )

    def q_value(self, norm_x, norm_y, norm_a):
        return np.dot(self.function_weights, self.q_feature(norm_x, norm_y, norm_a))


    def reset(self):
        super().reset()
        self.behavior_policy = np.random.random((self.rows, self.cols, len(self.action_space)))
        for x in range(self.rows):
            for y in range(self.rows):
                max_index = np.argmax(self.behavior_policy[x, y])
                self.behavior_policy[x, y] = self.epsilon_greedy(max_index, 0.5)

    def select_action_behavior(self, x, y):
        action_index = int(
            torch.distributions.Categorical(torch.FloatTensor(self.behavior_policy[x, y]))
            .sample()
            .item()
        )
        return self.action_space[action_index], action_index

    def update_values(self):
        for x in range(self.rows):
            for y in range(self.cols):
                nx, ny = self.normalize_coordinates(x, y)

                self.state_values[x, y] = self.q_value(nx, ny)
                with torch.no_grad():
                    self.policy[x, y] = self.target_policy_network(torch.tensor([nx, ny]).to(self.device)).cpu().detach().numpy()

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            losses = []
            x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
            a, ai= self.select_action_behavior(x, y)
            for _ in range(self.episode_length):
                r, (x2, y2) = self.get_next_state_and_reward((x, y), a)
                nx, ny = self.normalize_coordinates(x, y)
                nx2, ny2 = self.normalize_coordinates(x2, y2)
                # TD error
                delta = r + self.HP.gamma * self.v_value(nx2, ny2) - self.v_value(nx, ny)

                # Actor update
                pi_p = self.target_policy_network(torch.tensor([nx, ny]).to(self.device))[ai]
                ro_p = self.behavior_policy[x, y, ai]
                loss = -torch.log(pi_p) * delta * pi_p / (ro_p+ 1e-8)
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                losses.append(loss.item())

                # Critic update
                v = self.v_value(nx, ny)
                v2 = self.v_value(nx2, ny2)
                self.function_weights += self.lr * delta * (pi_p.item()/ self.behavior_policy[x, y][ai]) * self.v_feature(nx, ny)
                a2, ai2 = self.select_action_behavior(x2, y2)
                x, y = x2, y2
                a, ai = a2, ai2
                

            self.update_values()    
            self.draw_picture(1)
            iter.set_postfix({"loss": f"{np.mean(losses):.4f}"})

hp = HyperParameters()
hp.rows = 5
hp.cols = 5
A2C_ImportanceSampling(hp).train()