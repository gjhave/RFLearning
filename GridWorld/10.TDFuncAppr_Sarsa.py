"""action-value function approximation
Initialization: 
    Initial parameter w0. Initial policy π0. αt = α > 0 for all t. ∈ (0, 1).
Goal: 
    Learn an optimal policy that can lead the agent to the target state from an initial state s0.

For each episode, do
    Generate a0 at s0 following π0(s0)
    If st (t = 0, 1, 2, . . . ) is not the target state, do
        Collect the experience sample (rt+1, st+1, at+1) given (st, at):
            generate rt+1, st+1 by interacting with the environment; generate at+1 following πt(st+1).
        Update q-value:  
            wt+1 = wt + αt  [  rt+1 + γqˆ(st+1, at+1, wt) − qˆ(st, at, wt)  ]  ∇wqˆ(st, at, wt)
        Update policy:  
            πt+1(a|st) = 1 − ε  |A(st)| (|A(st)| − 1) if a = arg maxa∈A(st) qˆ(st, a, wt+1) πt+1(a|st) = |A(st)| otherwise
        st ← st+1, at ← at+1
"""

"""
函数近似方法相较于表格方法，就是把之前所有的算法中，直接从表格根据环境获取的状态-动作价值，用一个函数来逼近
v(s)=f(s, w)
q(s, a)=g(s, a, w)
这里w就是函数的权重，需要通过学习来更新
跟新方法就是用RM算法来更新w
这里，我们演示基于q(s, a, w)的Sarsa算法
w := w + lr * (r + gamma * q(s', a', w) - q(s, a, w))
其中r + gamma * q(s', a', w)是状态-动作价值的估计值，q(s, a, w)是当前状态-动作价值的采样值
这是在给定策略下通过更新w来逼近真是的状态-动作价值的函数
还需要跟新策略来达到最优策略
"""


import numpy as np
import tqdm
from GridWorld import GridWorldEnv, HyperParameters
import torch
import torch.nn as nn

class TDFuncAppro_Sarsa(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 1000
        self.lr = 0.001
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.network = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)
        self.lr = 0.0001
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.lr)

    def value_iteration_step(self):
        stable = True
        x, y = np.random.randint(0, self.rows), np.random.randint(
            0, self.cols
        )  # 随机选择一个状态进行更新
        a, ai = self.select_action(x, y)
        # x, y = 0, 0  # for debug
        for _ in range(self.episode_length):
            # self.agent_step(x, y)  # for debug

            # 获取下一个动作和下一个状态
            r, (x2, y2) = self.get_next_state_and_reward(
                (x, y), a
            )  # 执行动作并获得奖励和下一个状态
            a2, ai2 = self.select_action(x2, y2)  # 根据当前策略选择一个动作

            # 归一化
            na = self.normalize_action(ai)
            na2 = self.normalize_action(ai2)
            nx, ny = self.normalize_coordinates(x, y)
            nx2, ny2 = self.normalize_coordinates(x2, y2)


            # 计算状态-动作价值的目标值和估计值
            q_s = self.network(torch.tensor(np.array([nx, ny, na]), dtype=torch.float32, device = self.device))
            q_s1 = r + self.gamma* self.network(torch.tensor(np.array([nx2, ny2, na2]), dtype=torch.float32, device = self.device))

            # 更新权重
            # """w := w + lr * (r + gamma * q(s', a', w) - q(s, a, w)) * grad(q(s, a, w))"""
            # """这里q(s, a, w)对w的梯度，就是q(s, a, w)的特征向量"""
            loss = self.loss_fn(q_s, q_s1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            



            # 更新策略
            with torch.no_grad():
                qvs = []
                for a_index in range(len(self.action_space)):
                    n = self.normalize_action(a_index)
                    qvs.append(self.network(torch.tensor(np.array([nx, ny, n]), dtype=torch.float32, device = self.device)).item())
                self.policy[x, y] = self.epsilon_greedy(
                    np.argmax(qvs), epsilon=self.epsilon
                )
                sv = np.dot(self.policy[x, y], qvs)
                if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                    stable = False

                self.state_values[x, y] = sv

            x, y = x2, y2
            a, ai = a2, ai2
        return stable

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            state_value_stable = self.value_iteration_step()
            self.draw_picture(1)
            self.lr *= 0.999
            self.epsilon *= 0.999
            iter.set_description(f"lr={self.lr:.6f}, epsilon={self.epsilon:.6f}")

            if state_value_stable:
                print(f"\nPolicy iteration converged after {i + 1} iterations!")
                break

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.max_iterations = 10000
hp.gamma = 0.9
hp.rows = 5
hp.cols = 5
env = TDFuncAppro_Sarsa(hp, action_space=5)
env.train()
