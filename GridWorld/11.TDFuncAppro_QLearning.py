"""
Initialization: 
    Initial parameter w0. Initial policy π0. αt = α > 0 for all t. ∈ (0, 1). 
Goal: 
    Learn an optimal path that can lead the agent to the target state from an initial state s0.  
    
For each episode, do 
    If st (t = 0, 1, 2, . . . ) is not the target state, do 
        Collect the experience sample (at, rt+1, st+1) given st: 
            generate at following πt(st); generate rt+1, st+1 by interacting with the environment. 
        Update q-value:  
            wt+1 = wt + αt  [  rt+1 + γ max  a∈A(st+1) qˆ(st+1, a, wt) − qˆ(st, at, wt)  ]  ∇wqˆ(st, at, wt)  
        Update policy:  
            πt+1(a|st) = 1 − ε |A(st)| (|A(st)| − 1) if a = arg maxa∈A(st) qˆ(st, a, wt+1)  πt+1(a|st) = ε  |A(st)| otherwise
"""

import numpy as np
import tqdm
from GridWorld import GridWorldEnv, HyperParameters
import cv2
import torch
import torch.nn as nn

class TDFuncAppro_QLearning(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 500
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
        self.optim = torch.optim.Adam(self.network.parameters(), self.lr)
        self.loss_fn = nn.MSELoss()



    def Episode_step(self):
        """评估当前策略，更新状态价值"""
        policy_stable = True

        x, y = np.random.randint(0, self.rows), np.random.randint(
            0, self.cols
        )  # 随机选择一个状态进行更新
        losses = []
        for _ in range(self.episode_length):
            a, ai = self.select_action(x, y)  
            r, (x2, y2) = self.get_next_state_and_reward((x, y), a)

            # 归一化
            nx, ny = self.normalize_coordinates(x, y)
            na = self.normalize_action(ai)
            nx2, ny2 = self.normalize_coordinates(x2, y2)

            qvs1 = []
            # Q-learning在更新时使用max_a' Q(s', a')，因此需要计算下一个状态所有动作的价值
            # w下 x2, y2的值
            # with torch.no_grad():
            for wai, wa in enumerate(self.action_space):
                wna = self.normalize_action(wai)
                value = self.network(torch.tensor(np.array([nx2, ny2, wna]), dtype=torch.float32, device=self.device))
                qvs1.append(value)
            max_q1 = torch.max(torch.stack(qvs1))

            # update value
            q_s = self.network(torch.tensor(np.array([nx, ny, na]), dtype=torch.float32, device=self.device))
            q_s1 = r + self.gamma * max_q1
            loss = self.loss_fn(q_s[0], q_s1)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())

            # update policy
            qvs = []
            with torch.no_grad():
                for w1ai, w1a in enumerate(self.action_space):
                    w1na = self.normalize_action(w1ai)
                    value = self.network(torch.tensor(np.array([nx, ny, w1na]), dtype=torch.float32, device=self.device))
                    qvs.append(value)
            
                max_index = torch.argmax(torch.stack(qvs)).item()
            self.policy[x, y] = self.epsilon_greedy(max_index, self.epsilon)
                # 更新状态价值为下一个状态的最大价值
            sv = max_q1.item()
            if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                policy_stable = False
            self.state_values[x, y] = sv  # 更新状态价值为当前策略下的期望价值


            a, ai = self.select_action(x2, y2)
            x, y = x2, y2  # 移动到下一个状态估后进行一次策略改进
        return policy_stable, np.mean(losses)

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            policy_stable, loss = self.Episode_step()
            self.draw_picture(1)
            self.lr *= 0.999
            self.epsilon *= 0.999
            iter.set_description(f"lr={self.lr:.6f}, epsilon={self.epsilon:.6f}, ava_loss={np.mean(loss):.6f}")

            if policy_stable:
                print(f"\nPolicy iteration converged after {i + 1} iterations!")
                break

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.gamma = 0.5
hp.rows = 5
hp.cols = 5
env = TDFuncAppro_QLearning(hp, action_space=5)
env.train()
