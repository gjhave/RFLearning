"""REINFORCE
Initialization: 
    Initial parameter θ; γ ∈ (0, 1); α > 0.
Goal: 
    Learn an optimal policy for maximizing J(θ).

For each episode, do
    Generate an episode {s_{0}, a_{0}, r_{1}, . . . , s_{T-1}, a_{T-1}, r_{T} } following π(θ).
    For t = 0, 1, . . . , T-1:
        Value update: 
            q_{t}(st, at) = ∑^{T}_{k=t+1} γ^{k-t-1}*r_{k}
        Policy update: 
            θ ← θ + α∇_{θ}lnπ(a_{t}|s_{t}, θ)q_{t}(s_{t}, a_{t})
"""




"""
这是收个策略梯度算法，之前的策略都是先算价值函数，然后根据价值函数来更新策略的，
这个算法直接用一个神经网络来逼近策略函数，输入状态，输出动作的概率分布，然后根据采样的经验来更新这个神经网络的参数，使得它能够更好地逼近最优策略。
该策略梯度有个非常严重的问题，就是梯度=∇_{θ}lnπ(a_{t}|s_{t}, θ)q_{t}(s_{t}, a_{t})这里q值可能是负的，而
"""

from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm
import torch
import torch.nn as nn


class ReinInforce(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.train_epoch = 5000
        self.episode_length = 1000
        self.lr = 0.001
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )  # 使用MPS加速（如果可用）
        # 定义一个简单的神经网络来逼近策略函数
        self.policy_network = nn.Sequential(
            nn.Linear(2, 128),  # 输入状态（x, y）
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.action_space)),
            nn.Softmax(dim=-1),  # 输出动作概率分布
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.torch_action_space = torch.tensor(
            [self.normalize_action(i) for i in range(len(self.action_space))]
        ).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.target = None
        for x in range(self.rows):
            for y in range(self.cols):
                if self.grid[x, y] == self.target_flag:
                    self.target = [y, x]
                    break

    def get_episode(self):
        states = []
        actions = []
        rewards = []
        norm_states = []
        total_returns = []
        x, y = np.random.choice(self.rows), np.random.choice(self.cols)
        # next_x, next_y = 0, 0  # for debug

        for t in range(self.episode_length):
            # self.agent_step(next_x, next_y)  # for debug
            # if next_x == self.target[0] and next_y == self.target[1]:
            #     break
            states.append([x, y])
            nx, ny = self.normalize_coordinates(x, y)
            norm_states.append([nx, ny])
            # a, ai = self.select_action(x, y)
            with torch.no_grad():
                p = self.policy_network(torch.tensor([nx, ny], dtype=torch.float32, device=self.device))
                dist = torch.distributions.Categorical(p)
                ai = dist.sample()
            actions.append(ai.item())
            r, (x2, y2) = self.get_next_state_and_reward((x, y), self.action_space[int(ai.item())])
            rewards.append(r)
            x, y = x2, y2
        r = 0
        total_returns = self.compute_return(rewards)
        return states, norm_states, actions, rewards, total_returns

    def compute_return(self, rewards):
        """计算 qt(st, at) = Σ_{k=t+1}^T γ^{k-t-1} r_k"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def policy_improvement(self):
        for x in range(self.rows):
            for y in range(self.cols):
                norm_x, norm_y = self.normalize_coordinates(x, y)
                # self.action_values[x, y, :] = avs[x, y, :] / (vc[x, y, :] + 1e-8)
                self.policy[x, y] = (
                    self.policy_network(
                        torch.FloatTensor([norm_x, norm_y]).to(self.device)
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .flatten()
                )
                # self.state_values[x, y] = np.dot(
                #     self.policy[x, y], self.action_values[x, y]
                # )  # 更新状态价值为当前策略下的期望价值
        self.draw_picture(wait_time=1)  # 每次策略改进后更新图片

    def train(self):
        iter = tqdm.tqdm(range(self.train_epoch))
        for epoch in iter:
            sts, nsts, acs, rs, trs = self.get_episode()  # 生成一个episode

            avs = np.zeros_like(self.action_values)  # action values
            vc = np.zeros_like(self.action_values)  # vist count
            episode_loss = []
            # value update
            with torch.enable_grad():
                # policy update
                loss = torch.tensor(0, dtype=torch.float32).to(self.device)
                for t in range(len(sts)):
                    policy = self.policy_network(
                        torch.FloatTensor(nsts[t]).to(self.device)
                    )
                    # 获取实际动作的 log 概率
                    # action_tensor = torch.tensor([actions[t]]).to(self.device)
                    """这里一定要注意，是对当前状态的具体动作的log求导，即lnπ(a_{t}|s_{t}, θ)这里是有a_{t}的"""
                    log_prob = torch.log(policy[acs[t]] + 1e-8)  # 加小常数防止 log(0)
                    G = trs[t]
                    loss += -log_prob * G
                    episode_loss.append(loss.item())
                """MC方法不能每一步都更新梯度，每步都更新，偏差会很大，无法收敛"""
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                self.policy_improvement()
            iter.set_description(
                f"Epoch {epoch + 1}/{self.train_epoch}, Loss: {np.mean(episode_loss):.8f}"
            )
        self.draw_picture()


hp = HyperParameters()
hp.rows = 5
hp.cols = 5
ReinInforce(hp, action_space=9).train()
