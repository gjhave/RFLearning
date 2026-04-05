"""
蒙特卡罗方法的本质是通过大量采样来估计状态-动作对的价值（即均值估计），从而进行策略评估和改进。它既可以在value iteration的框架下使用，也可以在policy iteration的框架下使用。
因为state value和action value都是return值的期望/均值，所以在策略评估阶段，我们通过采样来估计每个状态-动作对的价值。
在策略改进阶段，和policy iteration/value iteration保持一致。
当在policy iteration中，采用的epsilon必须很小，否则很难收敛，原因是随机探索会引入高方差，导致价值估计不稳定，策略改进会反复震荡
"""

"""value iteration
Initialization: 
    Initial guess π_{0}.  
Goal: 
    Search for an optimal policy. 

For the kth iteration (k = 0, 1, 2, . . . ), do 
    For every state s ∈ S, do 
        For every action a ∈ A(s), do 
            Collect sufficiently many episodes starting from (s, a) by following π_{k} 
            Policy evaluation:  
                q_{π_{k}}(s, a) ≈ q_{k}(s, a) = the average return of all the episodes starting from (s, a)  
        Policy improvement:  
            a^{*}_{k}(s) = arg max_{a} q_{k}(s, a)  
            π_{k+1}(a|s) = 1 if a = a^{*}_{k}, and π_{k+1}(a|s) = 0 otherwise

"""

"""policy iteration
Initialization: 
    Initial guess π_{0}.  
Goal: 
    Search for an optimal policy.  

While v_{π} has not converged, for the kth iteration, do
    Policy evaluation:
        While v_{π_{k}} has not converged, for the kth iteration, do
            for every state s ∈ S, do  
                Collect sufficiently many episodes starting from (s, a) by following π_{k} 
                v_{π_{k}}(s) ≈ the average return of all the episodes starting from s by following π_{k}
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


class MentoCarioBsic(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.sample_episodes = 20  # 每个状态-动作对采样的episode数量，增加这个数量可以提高策略评估的准确性，但也会增加计算时间
        self.episode_length = 100  # 从改变episode长度的经验来看，过短的episode可能无法充分捕捉到环境的动态和奖励结构，导致策略评估不准确；最终迭代的policy会最优，但是state value不一定最优。


        """ 
        MC方法的特性：
        return = Rt+1 + γRt+2 + γ2Rt+3 + . . . .，epsiode越长，引入的随机变量越多，方差越大。
        这里epsilon必须足够小，否则很难收敛，原因是随机探索，epsiode越往后，越会引入高方差，导致价值估计不稳定，策略改进会反复震荡
        """
        self.truncate_num = 10  # 使用truncate policy iteration来加速收敛，即在策略评估阶段只迭代固定次数，而不是一直迭代到完全收敛，这样可以在某些情况下更快地找到一个近似最优的策略。

    def value_iteration_step(self):
        """执行一步值迭代"""
        stable = True  # 用于检查策略是否稳定
        for x in range(self.rows):
            for y in range(self.cols):
                for ai, a in enumerate(self.action_space):
                    qvs = []
                    # 从x, y出发，每个动作采样多条episode来评估当前策略的action value
                    for _ in range(self.sample_episodes):
                        rs = []
                        x2, y2 = x, y
                        a2 = a
                        for _ in range(self.episode_length):
                            r, (x2, y2) = self.get_next_state_and_reward((x2, y2), a2)
                            a2, _ = self.select_action(x2, y2)
                            rs.append(r)
                        qvs.append(sum([self.gamma**i * r for i, r in enumerate(rs)]))
                    # 取多条episode的平均return作为动作对action value的估计
                    self.action_values[x, y, ai] = np.mean(qvs)
                # policy update, 选择最大动作价值对应的动作作为当前状态的最优动作
                self.policy[x, y] = self.epsilon_greedy(
                    np.argmax(self.action_values[x, y]), epsilon=0
                )

                # update state value to max action value
                sv = max(self.action_values[x, y])  # 取最大动作价值作为状态价值
                if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                    stable = False  # 如果状态价值变化超过阈值，认为策略还没有达到最优
                self.state_values[x, y] = sv  # 更新状态价值
        return stable  # 返回状态值是否稳定

    def policy_evaluation(self):
        """这里采用采样均值的方式替代policy iteration方法中的收缩映射迭代的方法来计算state value"""
        stable = True
        for x in range(self.rows):
            for y in range(self.cols):
                gs = []  # 存储每个episode的return
                x2, y2 = x, y
                # 从x,y出发，每个动作采样多条episode来评估当前策略的action value
                for _ in range(self.sample_episodes):
                    rs = []
                    for _ in range(self.episode_length):
                        a, _ = self.select_action(x2, y2)  # 根据当前策略选择动作
                        r, (x2, y2) = self.get_next_state_and_reward((x2, y2), a)
                        rs.append(r)
                    # 计算当前episode的return
                    g = sum([self.gamma**i * r for i, r in enumerate(rs)])
                    gs.append(g)
                # 取多条episode的平均return作为状态价值的估计
                sv = np.mean(gs)
                if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                    stable = False  # 如果状态价值变化超过阈值，认为策略还没有达到最优
                self.state_values[x, y] = sv  # 更新状态价值
        return stable  # 返回状态值是否稳定

    def policy_improvement(self):
        """策略改进：对每个状态选择最优动作"""
        for x in range(self.rows):
            for y in range(self.cols):
                # 计算每个动作的价值
                qvs = []  # 存储每个动作的价值
                for a in self.action_space:
                    r, s2 = self.get_next_state_and_reward([x, y], a)
                    value = r + self.gamma * self.state_values[s2[0], s2[1]]
                    qvs.append(value)

                """ 
                return = Rt+1 + γRt+2 + γ2Rt+3 + . . . .，epsiode越长，引入的随机变量越多，方差越大。
                这里epsilon必须足够小，否则很难收敛，原因是随机探索，epsiode越往后，越会引入高方差，导致价值估计不稳定，策略改进会反复震荡
                """
                # 找出最优动作,并更新策略为在最优动作上选择概率为1，其他动作概率为0
                self.policy[x, y] = self.epsilon_greedy(
                    np.argmax(qvs),
                    epsilon=self.epsilon,
                )

    def train(self):

        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            """value iteration的策略评估和改进是交织在一起的，所以不需要单独的policy evaluation步骤"""
            # state_value_converged = self.value_iteration_step()
            # self.draw_picture(1)  # 每次迭代后绘制一次图像
            # if state_value_converged:
            #     print(f"\nPolicy iteration converged after {i + 1} iterations!")
            #     break

            """policy iteration的策略评估和改进是分开的，所以需要单独的policy evaluation步骤"""
            state_value_stable = self.policy_evaluation()
            # 策略改进
            self.policy_improvement()
            self.draw_picture(1)  # 每次评估后绘制一次图像
            if state_value_stable:
                print(f"\nPolicy iteration converged after {i + 1} iterations!")
                break

        self.print_optimal_policy()  # 输出最终的最优策略
        self.draw_picture()


hp = HyperParameters()
hp.gamma = 0.5
hp.rows = 5
hp.cols = 5
env = MentoCarioBsic(hp, action_space=5)
env.train()
