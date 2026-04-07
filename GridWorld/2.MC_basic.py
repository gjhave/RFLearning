"""
蒙特卡罗方法的本质是通过大量采样来估计状态-动作对的价值（即均值估计），从而进行策略评估和改进。
因为action value是return值的期望/均值，所以在策略评估阶段，我们通过大量采样来估计每个状态-动作对的价值。
在策略改进阶段，和policy iterationn保持一致。
当在policy iteration中，采用的epsilon必须很小，否则很难收敛，原因是随机探索会引入高方差，导致价值估计不稳定，策略改进会反复震荡
"""

"""policy iteration
Initialization: 
    Initial guess π0.  
Goal: 
    Search for an optimal policy.  
    
For the kth iteration (k = 0, 1, 2, . . . ), do 
    For every state s ∈ S, do 
        For every action a ∈ A(s), do 
            Collect sufficiently many episodes starting from (s, a) by following πk 
            Policy evaluation:  
                qπk(s, a) ≈ qk(s, a) = the average return of all the episodes starting from (s, a)  
        Policy improvement: 
            a∗k(s) = arg maxa qk(s, a)  πk+1(a|s) = 1 if a = a∗k, and πk+1(a|s) = 0 otherwise
"""

import numpy as np
import tqdm
from GridWorld import GridWorldEnv, HyperParameters


class MentoCarioBsic(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.sample_episodes = 50  # 每个状态-动作对采样的episode数量，增加这个数量可以提高策略评估的准确性，但也会增加计算时间
        self.episode_length = 100  # 从改变episode长度的经验来看，过短的episode可能无法充分捕捉到环境的动态和奖励结构，导致策略评估不准确；最终迭代的policy会最优，但是state value不一定最优。


        """ 
        MC方法的特性：
        return = Rt+1 + γRt+2 + γ2Rt+3 + . . . .，epsiode越长，引入的随机变量越多，方差越大。
        这里epsilon必须足够小，否则很难收敛，原因是随机探索，epsiode越往后，越会引入高方差，导致价值估计不稳定，策略改进会反复震荡
        """
        self.truncate_num = 10  # 使用truncate policy iteration来加速收敛，即在策略评估阶段只迭代固定次数，而不是一直迭代到完全收敛，这样可以在某些情况下更快地找到一个近似最优的策略。

    def sample_episdoe(self, x, y, a):
        rs = []
        x2, y2 = x, y
        a2 = a
        for _ in range(self.episode_length):
            r, (x2, y2) = self.get_next_state_and_reward((x2, y2), a2)
            a2, _ = self.select_action(x2, y2)
            rs.append(r)
        g = sum([self.gamma**i * r for i, r in enumerate(rs)])
        return g


    def policy_evaluation(self, x, y, a):
        gs = []  # 存储每个episode的return
        for _ in range(self.sample_episodes):
            g =self.sample_episdoe(x, y, a)
            gs.append(g)
        return np.mean(gs)

    def policy_improvement(self, x, y):
        max_index = np.argmax(self.action_values[x, y])
        self.policy[x, y] = self.epsilon_greedy(max_index, self.epsilon)

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            stable = True
            for x in range(self.rows):
                for y in range(self.cols):
                    for ai, a in enumerate(self.action_space):
                        q = self.policy_evaluation(x, y, a)
                        self.action_values[x, y, ai] = q
                    self.policy_improvement(x, y)
                    sv = np.max(self.action_values[x, y])
                    if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                        stable = False
                    self.state_values[x, y] = sv
            self.epsilon *= 0.5
            self.draw_picture(1)
            if stable:
               break



        self.print_optimal_policy()  # 输出最终的最优策略
        self.draw_picture()


hp = HyperParameters()
hp.gamma = 0.5
hp.rows = 5
hp.cols = 5
hp.epsilon = 0.5
env = MentoCarioBsic(hp, action_space=5)
env.train()
