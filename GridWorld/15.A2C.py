"""A2C
Initialization: 
    A policy function π(a|s, θ_{0}) where θ_{0} is the initial parameter. 
    A value function v(s, w_{0}) where w_{0} is the initial parameter. α_{w}, α_{θ} > 0. 
Goal: 
    Learn an optimal policy to maximize J(θ). 

At time step t in each episode, do 
    Generate at following π(a|s_{t}, θ_{t}) and then observe r_{t+1}, s_{t+1}. 
    Advantage (TD error):  
        δ_{t} = r_{t+1} + γv(s_{t+1}, w_{t}) − v(s_{t}, w_{t}) 
    Actor (policy update):  
        θ_{t+1} = θ_{t} + α_{θ}δ_{t}∇_{θ} ln π(a_{t}|s_{t}, θ_{t}) 
    Critic (value update):  
        w_{t+1} = w_{t} + α_{w}δ_{t}∇_{w}v(s_{t}, w_{t})
"""


from GridWorld import GridWorldEnv, HyperParameters
import torch
import torch.nn as nn
import tqdm
import numpy as np


class A2C(GridWorldEnv):
    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 1000
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.policy_network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.action_space)),
            nn.Softmax(dim=-1),
        ).to(self.device)
        self.lr = 0.001
        # v(s, w)是s的函数，对于s的坐标x,y有 v(s, w) = w[0]*x + w[1]*y + w[2]，这里我们用一个简单的线性函数逼近来估计状态-动作价值
        # 对于x,y,x*y,x^2,y^2等特征进行线性组合来逼近状态值，有v(s, w) = w[0]*x + w[1]*y + w[2]*x*y + w[3]*x^2 + w[4]*y^2 + w[5]，这里我们用一个简单的线性函数逼近来估计状态-动作价值
        self.function_weights = np.random.random(
            len(["x", "y", "x*y", "x^2", "y^2", "1"])
        )  # 线性函数逼近的权重，包含x,y,1三个特征的权重

        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)

    def v_feature(self, norm_x, norm_y):
        return np.array(
            [
                norm_x,
                norm_y,
                norm_x * norm_y,
                norm_x**2,
                norm_y**2,
                1,
            ]
        )

    def v_value(self, norm_x, norm_y):
        return np.dot(self.function_weights, self.v_feature(norm_x, norm_y))
    

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
            for _ in range(self.episode_length):
                nx, ny = self.normalize_coordinates(x, y)
                p = self.policy_network(torch.tensor([[nx, ny]]).to(self.device))
                dist = torch.distributions.Categorical(p)
                ai = dist.sample()
                a = self.action_space[int(ai.item())]
                r, (x2, y2) = self.get_next_state_and_reward((x, y), a)
                nx2, ny2 = self.normalize_coordinates(x2, y2)

                # TD error =q(s_{t}, a_{t}) - v(s_{t}) = r_{t+1} + γv(s_{t+1}, w_{t}) - v(s_{t}, w_{t})
                v = self.v_value(nx, ny)
                v2 = self.v_value(nx2, ny2)
                delta = r + self.gamma * v2 - v

                # Actor update
                actor_loss = -delta * torch.log(dist.log_prob(ai))
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                self.policy_optimizer.step()

                # Critic update
                self.function_weights += self.lr * delta * self.v_feature(nx, ny)
                x, y = x2, y2

            with torch.no_grad():
                for x in range(self.rows):
                    for y in range(self.cols):
                        self.state_values[x, y] = self.v_value(x, y)
                        nx, ny = self.normalize_coordinates(x, y)
                        p = self.policy_network(torch.tensor([[nx, ny]]).to(self.device))
                        self.policy[x, y] = p.detach().cpu().numpy()
                self.draw_picture(1)
                        

hp = HyperParameters()
hp.rows = 5
hp.cols = 5
hp.gamma=0.5
A2C(hp, action_space=5).train()