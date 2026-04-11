"""
Initialization: 
    A policy function π(a|s, θ_{0}) where θ_{0} is the initial parameter. 
    A value function q(s, a, w_{0}) where w_{0} is the initial parameter. α_{w}, α_{θ} > 0.
Goal: 
    Learn an optimal policy to maximize J(θ).
    
At time step t in each episode, do
    Generate at following π(a|s_{t}, θ_{t}), observe r_{t+1}, s_{t+1}, and and then generate a_{t+1} following π(a|s_{t+1}, θ_{t}).
    Actor (policy update):
        θ_{t+1} = θ_{t} + α_{θ}∇_{θ} ln π(a_{t}|s_{t}, θ_{t})q(s_{t}, a_{t}, w_{t})
    Critic (value update):
        w_{t+1} = w_{t} + α_{w}  [r_{t+1} + γ*q(s_{t+1}, a_{t+1}, w_{t+1})] − q(s_{t}, a_{t}, w_{t})]∇_{w}q(s_{t}, a_{t}, w_{t})
"""

from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm
import torch
import torch.nn as nn


class QAC(GridWorldEnv):
    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 1000
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.policy_network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.action_space)),
            nn.Softmax(dim=-1),
        ).to(self.device)

        self.policy_lr = 0.001
        self.value_lr = 0.01
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.policy_lr
        )
        lambda1 = lambda epoch: epoch // 3
        lambda2 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda=[lambda1, lambda2])

        self.q_weights = np.random.normal(
            0, 0.1, len(["x", "y", "x*y", "x^2", "y^2", "a", "a^2", "1"])
        )
 

    def q_feature(self, norm_x, norm_y, norm_a):
        """计算状态-动作价值的特征向量"""
        # return np.array([x, y, x * y, x**2, y**2, a, a**2, 1], dtype=float)
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
        """计算状态-动作价值"""
        return np.dot(
            self.q_weights,
            self.q_feature(norm_x, norm_y, norm_a),
        )

    def update_values(self):
        for x in range(self.rows):
            for y in range(self.cols):
                nx, ny = self.normalize_coordinates(x, y)
                qvs = []
                for ai, a in enumerate(self.action_space):
                    na = self.normalize_action(ai)
                    q = self.q_value(nx, ny, na)
                    qvs.append(q)
                self.state_values[x, y] = np.max(qvs)
                with torch.no_grad():
                    self.policy[x, y]= self.policy_network(torch.FloatTensor([nx, ny]).to(self.device)).cpu().detach().numpy()
        self.draw_picture(1)

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            x, y = np.random.randint(0, self.rows), np.random.randint(
                0, self.cols
            )
            nx, ny = self.normalize_coordinates(x, y)
            p = self.policy_network(torch.tensor(np.array([nx, ny]), dtype=torch.float32, device=self.device))
            dist = torch.distributions.Categorical(p)
            ai = dist.sample()
            na = self.normalize_action(ai.item())
            losses = []
            self.policy_optimizer.zero_grad()
            for _ in range(self.episode_length):
                # q = torch.tensor(self.q_value(nx, ny, na),dtype=torch.float32, device=self.device)
                q = self.q_value(nx, ny, na)
                log_p = torch.log(p[ai])
                loss_p = -log_p * q
                losses.append(loss_p.item())

                loss_p.backward()

                r, (x2, y2) = self.get_next_state_and_reward([x, y], self.action_space[int(ai.item())])
                nx2, ny2 = self.normalize_coordinates(x2, y2)
                p2 = self.policy_network(torch.FloatTensor([nx2, ny2]).to(self.device))
                dist2 = torch.distributions.Categorical(p2)
                ai2 = dist2.sample()
                na2 = self.normalize_action(ai2.item())

                qvs_t = r + self.gamma * self.q_value(nx2, ny2, na2)
                qvs_m = self.q_value(nx, ny, na)
                self.q_weights += self.value_lr * (qvs_t - qvs_m) * self.q_feature(nx, ny, na)
                x, y = x2, y2
                ai, na = ai2, na2
                nx, ny = nx2, ny2
                p = p2
            self.policy_optimizer.step()
            lr = self.scheduler.get_lr()
            self.update_values()
            iter.set_postfix({"loss": np.mean(losses)})

hp = HyperParameters()
hp.rows = 5
hp.cols = 5
QAC(hp, 5).train()