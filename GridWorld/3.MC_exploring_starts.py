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
        self.episode_length = 100  # 固定长度的episode
        self.lr = 0.999

    def get_episode(self, x=None, y=None, ai=None):
        """生成一个完整的episode，返回状态-动作-奖励序列"""
        states = []
        actions = []
        rewards = []
        # 从tartget状态开始，确保每个状态-动作对都有机会被探索
        if x is None and y is None and ai is None:
            x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
            a, ai = self.select_action(x, y)
        # x, y = 0, 0
        for _ in range(self.episode_length):
            # self.agent_step(x, y)
            states.append((x, y))  # 记录状态和动作
            actions.append(ai)
            r, (x2, y2) = self.get_next_state_and_reward((x, y), self.action_space[ai])
            a2, ai2 = self.select_action(x2, y2)
            x, y = x2, y2
            ai = ai2
            rewards.append(r)

        return states, actions, rewards

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        Returns = np.zeros((self.rows, self.cols, len(self.action_space)), dtype=np.float32)
        Nums = np.zeros_like(Returns, dtype=np.float32)
        for i in iter:
            states, actions, rewards = self.get_episode()
            g = 0
            for t in reversed(range(len(states))):
                x, y = states[t]
                ai = actions[t]
                r = rewards[t]
                g = self.gamma * g + r
                Returns[x, y, ai] += g
                Nums[x, y, ai] += 1

                # policy evaluation
                self.action_values[x, y, ai] = Returns[x, y, ai] / Nums[x, y, ai]

                # policy improvement
                max_index = np.argmax(self.action_values[x, y])
                self.policy[x, y] = self.epsilon_greedy(max_index, self.epsilon)
            
            self.epsilon *= self.lr

            self.draw_picture(1)
            iter.set_description(f"epsilon={self.epsilon:.4f}")


        self.print_optimal_policy()  # 输出最终的最优策略
        self.draw_picture()


hp = HyperParameters()
hp.rows = 5
hp.cols = 5
hp.gamma = 0.5
hp.epsilon = 0.1
hp.max_iterations = 10000

env = MCEpsilonGreedy(hp, action_space=5)
env.train()
