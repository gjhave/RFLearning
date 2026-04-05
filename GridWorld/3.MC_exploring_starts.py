"""
MC Exploring Starts算法的核心思想是在每个迭代中，只产生一个很长的episode，并且这个episode是从一个随机的状态-动作对开始的，这样可以确保所有状态-动作对都有机会被探索到很多次。
而MC Basic算法则是在每个迭代中，对每个状态-动作对都产生多个episode，这样可以更全面地评估每个状态-动作对的价值，但也会增加计算时间。
就model free方法来说，MC Exploring Starts算法对sample data的利用率更高。
"""

"""
Initialization: 
    Initial policy π_{0}(a|s) and initial value q(s, a) for all (s, a). 
    Returns(s, a) = 0 and Num(s, a) = 0 for all (s, a).
Goal: 
    Search for an optimal policy.

For each episode, do
    Episode generation:
        Select a starting state-action pair (s_{0}, a_{0}) and ensure that all pairs can be possibly selected (this is the exploring-starts condition).
        Following the current policy, generate an episode of length T : s_{0}, a_{0}, r_{1}, . . . , s_{T-1}, a_{T-1}, r_{T}.

    Initialization for each episode: g ← 0
    For each step of the episode, t = T-1, T-2, . . . , 0, do
        g ← γ*g + r_{t+1}
        Returns(s_{t}, a_{t}) ← Returns(s_{t}, a_{t}) + g
        Num(s_{t}, a_{t}) ← Num(s_{t}, a_{t}) + 1

        Policy evaluation:
            q(s_{t}, a_{t}) ← Returns(s_{t}, a_{t})/Num(s_{t}, a_{t})

        Policy improvement:
            π(a|s_{t}) = 1 if a = arg max_a q(s_{t}, a) and π(a|s_{t}) = 0 otherwise
"""


import numpy as np
import tqdm
from GridWorld import GridWorldEnv, HyperParameters


class MCEpsilonGreedy(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)


        """ 
        MC方法的特性：
        return = Rt+1 + γRt+2 + γ2Rt+3 + . . . .，epsiode越长，引入的随机变量越多，方差越大。
        这里epsilon必须足够小，否则很难收敛，原因是随机探索，epsiode越往后，越会引入高方差，导致价值估计不稳定，策略改进会反复震荡
        """
        # self.episode_length = int(
        #     self.rows * self.cols * 200 * len(self.action_space)
        # )  # 根据网格大小和动作数量动态调整episode长度
        self.episode_length = 100000  # 固定长度的episode

    def reset(self):
        super().reset()
        """重置环境但保持网格不变"""

    def get_episode(self):
        """生成一个完整的episode，返回状态-动作-奖励序列"""
        states = []
        actions = []
        rewards = []
        # 从tartget状态开始，确保每个状态-动作对都有机会被探索

        x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
        for _ in range(self.episode_length):
            states.append((x, y))  # 记录状态和动作
            a, ai = self.select_action(x, y)
            actions.append(ai)
            reward, (x, y) = self.get_next_state_and_reward((x, y), a)
            rewards.append(reward)

        return states, actions, rewards

    def value_iteration_step(self):
        """执行MC Exploring Starts算法"""
        stable = True
        states, actions, rewards = self.get_episode()  # 第一次迭代使用完全随机的episode
        visit_count = np.zeros(
            (self.rows, self.cols, len(self.action_space)), dtype=int
        )  # 每个状态-动作对的访问次数
        returns = np.zeros(
            (self.rows, self.cols, len(self.action_space)), dtype=np.float32
        )
        g = 0
        for t in reversed(range(len(states))):
            x, y = states[t]
            ai = actions[t]
            r = rewards[t]
            g = self.gamma * g + r  # 计算从t时刻开始的回报
            returns[x, y, ai] += g  # 每访问一次某个(s, a)，增加一个sample，
            visit_count[x, y, ai] += 1  # 记录访问该(s, a)的次数
            # policy evaluation, 使用均值估计方法来更新action value
            self.action_values[x, y, ai] = (
                returns[x, y, ai] / (visit_count[x, y, ai]) + 1e-10
            )

            policy = self.epsilon_greedy(
                np.argmax(self.action_values[x, y]), epsilon=self.epsilon
            )
            self.policy[x, y] = policy
            # sv = np.dot(self.action_values[x, y], self.policy[x, y])
            sv = np.max(self.action_values[x, y])
            if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                stable = False  # 如果状态价值变化超过阈值，认为策略还没有达到最优
            self.state_values[x, y] = sv  # 更新状态价值

        return stable

    def policy_evaluation(self):
        # for x in range(self.rows):
        #     for y in range(self.cols):
        states, actions, rewards = self.get_episode()  # 第一次迭代使用完全随机的episode

        visit_count = np.zeros(
            (self.rows, self.cols, len(self.action_space)), dtype=int
        )  # 每个状态-动作对的访问次数
        returns = np.zeros(
            (self.rows, self.cols, len(self.action_space)), dtype=np.float32
        )
        g = 0
        for t in reversed(range(len(states))):
            x, y = states[t]
            ai = actions[t]
            r = rewards[t]
            g = self.gamma * g + r  # 计算从t时刻开始的回报
            returns[x, y, ai] += g  # 每访问一次某个(s, a)，增加一个sample，
            visit_count[x, y, ai] += 1  # 记录访问该(s, a)的次数
            # policy evaluation, 使用均值估计方法来更新action value
            self.action_values[x, y, ai] = returns[x, y, ai] / (
                visit_count[x, y, ai] + 1e-10
            )

    def policy_improvement(self):
        stable = True
        for x in range(self.rows):
            for y in range(self.cols):
                self.policy[x, y] = self.epsilon_greedy(
                    np.argmax(self.action_values[x, y]), epsilon=self.epsilon
                )
                # sv = np.dot(self.action_values[x, y], self.policy[x, y])
                sv = np.max(self.action_values[x, y])
                if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                    stable = False  # 如果状态价值变化超过阈值，认为策略还没有达到最优
                self.state_values[x, y] = sv  # 更新状态价值
        return stable

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            """value iteration 形式的MC Exploring Starts算法"""
            state_value_stable = self.value_iteration_step()
            self.draw_picture(1)  # 每次迭代后绘制一次图
            if state_value_stable:
                print(f"Policy converged after {i+1} iterations.")
                break

            """policy iteration 形式的MC Exploring Starts算法"""
            # self.policy_evaluation()
            # state_value_stable = self.policy_improvement()
            # self.draw_picture(1)  # 每次迭代后绘制一次图
            # if state_value_stable:
            #     print(f"Policy converged after {i+1} iterations.")
            #     break

        self.print_optimal_policy()  # 输出最终的最优策略
        self.draw_picture()


hp = HyperParameters()
hp.rows = 5
hp.cols = 5
hp.gamma = 0.5
hp.max_iterations = 1000

env = MCEpsilonGreedy(hp, action_space=5)
env.train()
