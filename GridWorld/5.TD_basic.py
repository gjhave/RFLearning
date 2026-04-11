"""
TD相对于MC方法的重要改进是增量式的迭代方法估计。
将用RM方法替代MC
不用等待所有的样本采集完成后再开始计算，而是来一个样本就估计一次。
本质就是用RM求v(s)
w_{k+1} = w_{k} - (w_{k} - x_{k})/k
w_{k}为第k次估计，x_{k}为第k次采样

"""

from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm


class TDBasic(GridWorldEnv):
    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 500
        self.lr = 0.001


    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
            stable = True
            for _ in range(self.episode_length):
                a, ai = self.select_action(x, y)
                r, (x2, y2) = self.get_next_state_and_reward((x, y), a)
                sv = self.state_values[x, y] - self.lr * (
                    self.state_values[x, y] - (r + self.gamma * self.state_values[x2, y2])
                )
                if np.abs(sv - self.state_values[x, y]) > self.HP.end_condition:
                    stable = False
                self.state_values[x, y] = sv
            self.draw_picture(1)
            self.lr *=0.99
            if stable:
                break

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.rows = 5
hp.cols = 5
hp.max_iterations = 100000
env = TDBasic(hp, action_space=5)
env.train()
