from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm


class Sarsa_Nstep(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 100
        self.lr = 0.001

    def policy_evaluation(self, n=5):
        """评估当前策略，更新状态价值"""
        stable = True
        # 随机选择一个状态进行更新
        x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
        for _ in range(self.episode_length):
            a, ai = self.select_action(x, y) 
            r, (x2, y2) = self.get_next_state_and_reward((x, y), a)

            # n-step
            for step in range(n - 1):
                a2, ai2 = self.select_action(x2, y2)
                r2, (x2, y2) = self.get_next_state_and_reward((x2, y2), a2)
                r += (self.gamma ** (step + 1)) * r2  # 累积n步的奖励

            a2, ai2 = self.select_action(x2, y2)

            # value update
            qv = self.action_values[x, y, ai] - self.lr * (
                self.action_values[x, y, ai]
                - (r + self.gamma * self.action_values[x2, y2, ai2])
            )
            if np.abs(qv - self.action_values[x, y, ai]) > self.HP.end_condition:
                stable = False
            self.action_values[x, y, ai] = qv

            # policy update
            self.policy_improvement(x, y)
            x, y = x2, y2  # 移动到下一个状态

        return stable

    def policy_improvement(self, x, y):
        """策略改进：为当前时刻状态选择最优动作"""
        # 计算每个动作的价值
        qvs = []
        for a in self.action_space:
            r, s2 = self.get_next_state_and_reward([x, y], a)
            q = r + self.gamma * self.state_values[s2[0], s2[1]]
            qvs.append(q)
        self.policy[x, y] = self.epsilon_greedy(np.argmax(qvs), self.epsilon)
        self.state_values[x, y] = sum(self.policy[x, y] * qvs)

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            policy_stable = self.policy_evaluation(n=5)
            if policy_stable:
                print(f"Policy converged after {i+1} iterations.")
                break
            self.lr *= 0.99
            self.epsilon *= 0.99
            iter.set_description(f"lr={self.lr:.6f}, epsilon={self.epsilon:.6f}")
            self.draw_picture(1)

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.gamma = 0.9
hp.rows = 5
hp.cols = 5
hp.max_iterations = 100000
env = Sarsa_Nstep(hp, action_space=5)
env.train()
