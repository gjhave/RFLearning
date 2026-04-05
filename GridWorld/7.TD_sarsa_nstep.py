from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm


class Sarsa_Nstep(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 100000
        self.lr = 0.001

    def policy_evaluation(self, n=5):
        """评估当前策略，更新状态价值"""
        stable = True
        # 随机选择一个状态进行更新
        x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
        for _ in range(self.episode_length):
            a, ai = self.select_action(x, y)  # 根据当前策略选择一个动作
            # 执行动作并获得奖励和下一个状态
            r, (x2, y2) = self.get_next_state_and_reward((x, y), a)

            for step in range(n - 1):
                a2, ai2 = self.select_action(x2, y2)  # 根据当前策略选择下一个状态的动作
                # 执行动作并获得奖励和下一个状态
                r2, (x2, y2) = self.get_next_state_and_reward((x2, y2), a2)
                r += (self.gamma ** (step + 1)) * r2  # 累积n步的奖励
            a2, ai2 = self.select_action(x2, y2)  # 根据当前策略选择下一个状态的动作
            # TD(0)更新：Q(s, a) = Q(s, a) + alpha * ( Q(s, a) - reward + gamma * Q(s', a'))
            qv = self.action_values[x, y, ai] - self.lr * (
                self.action_values[x, y, ai]
                - (r + self.gamma * self.action_values[x2, y2, ai2])
            )
            if np.abs(qv - self.action_values[x, y, ai]) > self.HP.end_condition:
                stable = False
            self.action_values[x, y, ai] = qv
            self.policy_improvement(x, y)
            x, y = x2, y2  # 移动到下一个状态

        return stable

    def policy_improvement(self, x, y):
        """策略改进：对每个状态选择最优动作"""
        # 计算每个动作的价值
        qvs = []
        for a in self.action_space:
            r, s2 = self.get_next_state_and_reward([x, y], a)
            q = r + self.gamma * self.state_values[s2[0], s2[1]]
            qvs.append(q)
        self.policy[x, y] = self.epsilon_greedy(np.argmax(qvs), self.epsilon)
        # 更新状态价值为当前策略下的期望价值
        self.state_values[x, y] = sum(self.policy[x, y] * qvs)
        self.draw_picture(wait_time=1)  # 每次策略改进后更新图片显示当前状态价值和策略

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            policy_stable = self.policy_evaluation(n=50)
            if policy_stable:
                print(f"Policy converged after {i+1} iterations.")
                break

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.gamma = 0.9
hp.rows = 5
hp.cols = 5
env = Sarsa_Nstep(hp, action_space=9)
env.train()
