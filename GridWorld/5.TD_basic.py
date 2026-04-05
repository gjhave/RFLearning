"""
TD相对于MC方法的重要改进是增量式的迭代方法估计。
将用RM、SGD等方法替代MC
不用等待所有的样本采集完成后再开始计算，而是来一个样本就估计一次。
本质就是用SGD求q(s,a)的估计值
w_{k+1} = w_{k} - (w_{k} - x_{k})/k
w_{k}为第k次估计，x_{k}为第k次采样

从RM到TD的变化过程
把x换成v(s)变成w_{t+1} = w_{t} - α * (w_{t} - v(s))
在把v(s)换成 r + γ*v(s)变成 w_{t+1} = w_{t} - α * (w_{t} - (r + γ*v(s)))
"""

from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm


class TDBasic(GridWorldEnv):
    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 500
        self.lr = 0.001


    def get_episode(self):
        states = []
        rewards = []
        for _ in range(self.episode_length):
            x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
            states.append((x, y))
            a, _ = self.select_action(x, y)
            r, (x2, y2) = self.get_next_state_and_reward((x, y), a)
            rewards.append(r)
        return states, rewards

    def policy_evaluation(self):
        x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
        for t in range(self.episode_length):
            a, ai = self.select_action(x, y)
            r, (x2, y2) = self.get_next_state_and_reward((x, y), a)
            sv_s1_t = self.state_values[x2, y2]
            sv_s_t = self.state_values[x, y]
            # v_{t+1}(s_{t}) = v_{t}(s_{t}) - α_{t}(s_{t})[v_{t}(s_{t}) - [r_{t+1}+γ*v_{t}(s_{t+1})]]
            sv_s_t1 = sv_s_t - self.lr * (sv_s_t - (r + self.gamma * sv_s1_t))
            self.state_values[x, y] = sv_s_t1
            x, y = x2, y2

    def policy_improvement(self):
        stable = True
        for x in range(self.rows):
            for y in range(self.cols):
                # 计算每个动作的价值
                qvs = []
                for ai, a in enumerate(self.action_space):
                    r, s2 = self.get_next_state_and_reward([x, y], a)
                    value = r + self.gamma * self.state_values[s2[0], s2[1]]
                    qvs.append(value)
                    if (
                        np.abs(self.action_values[x, y, ai] - value)
                        > self.HP.end_condition
                    ):
                        stable = False
                    self.action_values[x, y, ai] = value

                # 找出最优动作,并更新策略为在最优动作上选择概率为1，其他动作概率为0
                self.policy[x, y] = self.epsilon_greedy(
                    np.argmax(qvs),
                    epsilon=self.epsilon,  # 这里epsilon的值的大小会影响最终计算state value的大小
                )
        return stable  # 返回策略是否稳定

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            self.draw_picture(1)

            if policy_stable:
                print(f"Policy converged after {i+1} iterations.")
                break

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.rows = 5
hp.cols = 5
env = TDBasic(hp, action_space=5)
env.train()
