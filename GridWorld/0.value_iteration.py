"""
值迭代的过程，是由一个随机初始化的state_value计算出来的，所以中间过程并不代表当前策略下的state value，只是迭代中产生的中间值，没有任何意义
书中给出的伪代码，和第(3)文中的思路略为有点不同，但并不影响结果：
常规的思路是，先将v迭代收敛到v*，然后再用greedy更新策略到optimal。
伪代码中是在每一次迭代过程中，都用greedy跟新，但这并不影响结果。因为值迭代算法是解BOE的过程，在迭代的过程中，要计算每一个q后取最大值，因此不受当前policy影响。在每次迭代中跟新和最后更新都没有问题。
"""

"""
Initialization:
    The probability models p(r|s, a) and p(s'|s, a) for all (s, a) are known. 
    Initial guess v0.
Goal: 
    Search for the optimal state value and an optimal policy for solving the Bellman optimality equation.

While v_{k} has not converged in the sense that ‖v_{k} - v_{k-1}‖ is greater than a predefined small threshold, for the kth iteration, do
    For every state s ∈ S, do
        For every action a ∈ A(s), do
            q-value: q_{k}(s, a) = ∑_{r} p(r|s, a)*r + γ*∑_{s'} p(s'|s, a)*v_{k}(s')
        Maximum action value: a^{s}_{k}(s) = arg max_{a} q_{k}(s, a)
        Policy update: 
            π_{k+1}(a|s) = 1 if a = a^{*}_{k}, and π_{k+1}(a|s) = 0 otherwise
        Value update: 
            v_{k+1}(s) = max_{a} q_{k}(s, a)
"""

import numpy as np
import tqdm

from GridWorld import GridWorldEnv, HyperParameters


class ValueIteration(GridWorldEnv):
    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)

    def value_iteration_step(self):
        """执行一步值迭代"""
        stable = True  # 检查state value是否收敛到optimal state value
        for x in range(self.rows):
            for y in range(self.cols):

                # 计算所有动作的价值
                qvs = []
                for a in self.action_space:
                    r, s2 = self.get_next_state_and_reward([x, y], a)
                    # 计算return value
                    g = r + self.gamma * self.state_values[s2[0], s2[1]]
                    qvs.append(g)
                """在迭代中，每一步都更新当前state的policy"""
                # policy update, 选择最大动作价值对应的动作作为当前状态的最优动作
                self.policy[x, y] = self.epsilon_greedy(np.argmax(qvs), epsilon=0)
                # update state value to max action value
                sv = max(qvs)
                if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                    stable = False  # 如果状态价值变化超过阈值，认为策略还没有达到最优
                self.state_values[x, y] = sv  # 更新状态价值
        
        return stable  # 返回策略是否稳定

    def train(self):

        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            stable = self.value_iteration_step()
            self.draw_picture(1)  # 每次迭代后绘制一次图像

            # 检查收敛
            if stable:
                """在v(s)收敛到optimal后更新policy"""
                # for x in range(self.rows):
                #     for y in range(self.cols):
                #         qvs = []
                #         for a in self.action_space:
                #             r, s2 = self.get_next_state_and_reward([x, y], a)
                #             # 计算return value
                #             g = r + self.gamma * self.state_values[s2[0], s2[1]]
                #             qvs.append(g)
                #         ai = np.argmax(qvs)
                #         self.policy[x, y] = self.epsilon_greedy(ai, 0)
                print(f"\nConverged after {i + 1} iterations!")
                break

        self.print_optimal_policy()  # 输出最终的最优策略
        self.draw_picture()


hp = HyperParameters()
hp.gamma = 0.5
hp.rows = 5
hp.cols = 5
env = ValueIteration(hp, action_space=5)
env.train()
