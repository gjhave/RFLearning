"""
值迭代没有显示的固定某个策略，完全评估该策略所有state下的价值，然后再更新，而是在每个state都去更新当前state的策略和状态值，
而策略迭代则是先固定策略，更新状态值（用收缩映射迭代的方法解贝尔曼方程），直到状态值收敛后再去更新策略。
在policy evaluation过程中，每个state下的state value是采用迭代办法产生的，中间值没有任何意义，但是最后收敛值是当前策略下的state value
"""

"""
Initialization: 
    The system model, p(r|s, a) and p(s'|s, a) for all (s, a), is known. Initial guess π0.
Goal: 
    Search for the optimal state value and an optimal policy.

While v_{π_{k}} has not converged, for the kth iteration, do
    Policy evaluation:
        Initialization: an arbitrary initial guess v^{(0)}_{π_{k}} for the state value function.
        While v^{(j)}_{π_{k}} has not converged, for the jth iteration, do
            For every state s ∈ S, do
                v^{(j+1)}_{π_{k}} (s) = ∑_{a} π_{k}(a|s)*[∑_{r} p(r|s, a)*r + γ ∑_{s'} p(s'|s, a)*v^{(j)}_{π_{k}} (s')]

    Policy improvement:
        For every state s ∈ S, do
            For every action a ∈ A, do
                q_{π_{k}} (s, a) = ∑_{r} p(r|s, a)*r + γ ∑_{s'} p(s'|s, a)*v_{π_{k}} (s')
            a^{*}_{k}(s) = arg max_{a} q_{π_{k}} (s, a)
            π_{k+1}(a|s) = 1 if a = a^{*}_{k}, and π_{k+1}(a|s) = 0 otherwise
"""


import numpy as np
import tqdm
from GridWorld import GridWorldEnv, HyperParameters


class PolicyIteration(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.truncate_num = 10  # 使用truncate policy iteration来加速收敛，即在策略评估阶段只迭代固定次数，而不是一直迭代到完全收敛，这样可以在某些情况下更快地找到一个近似最优的策略。


    def policy_evaluation(self, truncate=False):
        """使用收缩映射迭代的方法进行策略评估"""
        iter_count = 0
        while (truncate and iter_count < self.truncate_num) or (not truncate):
            stable = True
            for x in range(self.rows):
                for y in range(self.cols):
                    qvs = []
                    for a in self.action_space:
                        r, s2 = self.get_next_state_and_reward([x, y], a)
                        g = r + self.gamma * self.state_values[s2[0], s2[1]]
                        qvs.append(g)
                    sv = np.dot(self.policy[x, y], qvs)
                    if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                        stable = False

                    self.state_values[x, y] = sv
            iter_count += 1
            if stable:
                print(f"\nState value converged after {iter_count} iterations!")
                break

    def policy_improvement(self):
        """策略改进：对每个状态选择最优动作"""
        stable = True
        for x in range(self.rows):
            for y in range(self.cols):
                # 计算每个动作的价值
                qvs = []
                for a in self.action_space:
                    r, s2 = self.get_next_state_and_reward([x, y], a)
                    g = r + self.gamma * self.state_values[s2[0], s2[1]]
                    qvs.append(g)

                # 找出最优动作,并更新策略为在最优动作上选择概率为1，其他动作概率为0
                policy = self.epsilon_greedy(np.argmax(qvs), epsilon=self.epsilon)
                if np.max(np.abs(policy - self.policy[x, y])) > self.HP.end_condition:
                    stable = False  # 如果策略变化超过阈值，认为策略还没有达到最优
                self.policy[x, y] = policy  # 更新策略
        return stable  # 返回策略是否稳定

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            # 策略评估# 使用truncate policy iteration来加速收敛
            self.policy_evaluation(truncate=True)

            # 策略改进
            policy_stable = self.policy_improvement()
            self.draw_picture(1)

            if policy_stable:
                print(f"\nPolicy converged after {i + 1} iterations!")
                break

        self.print_optimal_policy()  # 输出最终的最优策略
        self.draw_picture()


hp = HyperParameters()
hp.rows = 5
hp.cols = 5
hp.gamma = 0.3

env = PolicyIteration(hp, action_space=5)
env.train()
