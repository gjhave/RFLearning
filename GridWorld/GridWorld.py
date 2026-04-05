"""
作为grideworld环境的基类，定义了环境的基本属性和方法，包括网格布局、状态价值、策略表示、动作选择、状态转移和奖励计算等。具体的强化学习算法（如策略迭代、值迭代、蒙特卡罗方法等）将继承这个基类，并实现train方法来进行学习和优化策略。
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
import torch



class HyperParameters:
    r_forbidden = -1
    r_target = 1
    r_step = 0
    r_border = -1
    rows = 10
    cols = 10
    action_space = [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [0, 0],
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1],
    ]  # up, down, left, right, stay, up-left, down-right, up-right, down-left
    action_str = ["↑", "↓", "←", "→", "●", "↖", "↘", "↗", "↙"]
    gamma = 0.9
    max_iterations = 1000
    end_condition = 1e-6
    epsilon = 0.5


class GridWorldEnv(ABC):

    forbidden_flag = -1
    target_flag = 1

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        self.HP = HP
        self.rows = HP.rows
        self.cols = HP.cols
        self.gamma = HP.gamma
        self.epsilon = HP.epsilon
        assert action_space <= len(
            HP.action_space
        ), "Action space size cannot exceed the predefined action space in HyperParameters"
        self.action_space = HP.action_space[:action_space]
        self.action_str = HP.action_str[:action_space]
        self.grid_size = 100
        self.agent_size = 50
        self.agent_img = cv2.imread("agent.jpg")
        self.agent_img = cv2.resize(self.agent_img, (self.agent_size, self.agent_size))

        self.reset()  # 初始化状态价值和策略

        try:
            grid = np.load(
                f"grid_{self.rows}x{self.cols}.npy"
            )  # 加载之前保存的网格布局
        except FileNotFoundError:
            grid = None  # 如果没有找到文件，则创建新的网格布局

        if grid is not None and not newgrid:
            self.grid = grid
        else:
            self.create_grid()

        self.img = np.zeros(
            (self.rows * self.grid_size, self.cols * self.grid_size, 3), dtype=np.uint8
        )

        for x in range(self.rows):
            cv2.line(
                self.img,
                (0, x * self.grid_size),
                (self.cols * self.grid_size, x * self.grid_size),
                (0, 0, 0),
                2,
            )
            for y in range(self.cols):
                cv2.line(
                    self.img,
                    (y * self.grid_size, 0),
                    (y * self.grid_size, self.rows * self.grid_size),
                    (0, 0, 0),
                    2,
                )

                if self.grid[x, y] == self.forbidden_flag:
                    color = (0, 0, 255)  # Red for forbidden
                elif self.grid[x, y] == self.target_flag:
                    color = (0, 255, 0)  # Green for target
                else:
                    color = (255, 255, 255)  # White for empty cells

                cv2.rectangle(
                    self.img,
                    (y * self.grid_size, x * self.grid_size),
                    ((y + 1) * self.grid_size, (x + 1) * self.grid_size),
                    color,
                    -1,
                )
                # center_x, center_y = y * self.grid_size + 50, x * self.grid_size + 50

    def reset(self):
        self.optimal_policy = None
        self.policy = np.zeros(
            (self.rows, self.cols, len(self.action_space)), dtype=float
        )  # 每个状态的策略，表示为每个动作的概率分布
        self.policy[:, :, :] = 1 / len(self.action_space)  # 初始化为均匀随机策略
        self.state_values = np.zeros((self.rows, self.cols), dtype=float)
        self.action_values = np.zeros(
            (self.rows, self.cols, len(self.action_space)), dtype=float
        )  # 每个状态-动作对的价值

    def create_grid(self):
        """创建一个新的网格环境，随机放置目标和禁区"""
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

        # 设置目标位置（唯一真正的终止状态）
        target_loc = (np.random.randint(0, self.rows), np.random.randint(0, self.cols))
        self.target_loc = target_loc

        # 设置禁区（可进入，但有惩罚）
        num_forbidden = int(self.rows * self.cols * 0.2)
        for _ in range(num_forbidden):
            forbidden_loc = (
                np.random.randint(0, self.rows),
                np.random.randint(0, self.cols),
            )
            if forbidden_loc != target_loc:  # 确保目标位置不被设为禁区
                self.grid[forbidden_loc] = self.forbidden_flag

        self.grid[target_loc] = self.target_flag
        np.save(
            f"grid_{self.rows}x{self.cols}.npy", self.grid
        )  # 保存网格布局以便后续使用

    def select_action(self, x, y):
        """
        根据当前策略选择一个动作
        Args:
            x: 当前状态的x坐标
            y: 当前状态的y坐标
        Returns:
            action: 选择的动作
            action_index: 选择的动作在动作空间中的索引
        """
        # 这里改为torch来采样，主要是因为通过torch.softmax产生的概率分布，经常遇到sum(pob)=1.00000000001或者0.9999999999的情况。
        action_index = int(
            torch.distributions.Categorical(torch.FloatTensor(self.policy[x, y]))
            .sample()
            .item()
        )  # 根据当前策略的概率分布选择动作索引
        # prob = self.policy[x, y]
        # if np.abs(np.sum(prob) - 1.0) > 1e-9:  # 检查概率分布是否有效
        #     # print(
        #     #     f"Warning: Policy{x, y} probabilities do not sum to 1, sum={np.sum(prob)}"
        #     # )
        #     prob = prob / np.sum(prob)  # 归一化概率分布
        # action_index = np.random.choice(
        #     len(self.action_space), p=prob
        # )  # 根据概率分布选择动作索引
        return self.action_space[action_index], action_index

    def get_next_state_and_reward(self, current_loc, action):
        """
        根据当前状态和动作计算下一个状态和奖励
        Args:
            current_loc: 当前状态的坐标 (x, y)
            action: 选择的动作
        Returns:
            reward: 执行动作后得到的奖励
            next_loc: 执行动作后到达的下一个状态的坐标 (x, y)
        """
        next_loc = [current_loc[0] + action[0], current_loc[1] + action[1]]

        # 边界检查
        if (
            next_loc[0] < 0
            or next_loc[0] >= self.rows
            or next_loc[1] < 0
            or next_loc[1] >= self.cols
        ):
            return self.HP.r_border, current_loc  # 撞墙，留在原地

        # 根据格子类型返回奖励
        if self.grid[next_loc[0], next_loc[1]] == self.forbidden_flag:
            return self.HP.r_forbidden, next_loc  # 可以进入禁区，但得到负奖励
        elif self.grid[next_loc[0], next_loc[1]] == self.target_flag:
            return self.HP.r_target, next_loc  # 到达目标
        else:
            return self.HP.r_step, next_loc  # 普通格子

    def epsilon_greedy(self, max_action_index, epsilon=0.0):
        """epsilon-greedy策略选择动作, 适用于grideworld环境"""
        assert 0 <= epsilon < 1, "Epsilon must be in [0, 1)"
        policy = np.zeros(len(self.action_space))

        policy[max_action_index] = (
            1
            if epsilon == 0
            else 1 - epsilon * (len(self.action_space) - 1) / len(self.action_space)
        )
        policy[policy == 0] = (
            0 if epsilon == 0 else epsilon / len(self.action_space)
        )  # 其他动作以epsilon的概率选择
        return policy

    def best_policy_str(self):
        policy_str = np.empty((self.rows, self.cols), dtype=object)
        best_action_idx = np.argmax(self.policy, axis=-1)
        for action_idx in range(len(self.action_space)):
            policy_str[best_action_idx == action_idx] = self.action_str[action_idx]
        return policy_str

    def normalize_coordinates(self, x, y):
        """将坐标归一化到[-1, 1]/[0, 1]范围内"""
        # norm_x = (x - self.rows / 2) / (self.rows / 2)
        # norm_y = (y - self.cols / 2) / (self.cols / 2)
        norm_x = x / (self.rows - 1)  # 将x坐标归一化到[0, 1]范围内
        norm_y = y / (self.cols - 1)  # 将y坐标归一化到[0, 1]范围内
        return norm_x, norm_y

    def normalize_action(self, action_index):
        """将动作归一化到[-1, 1]/[0, 1]范围内"""
        # norm_a = (action_index - len(self.action_space) / 2) / (
        #     len(self.action_space) / 2
        # )
        norm_a = action_index / (
            len(self.action_space) - 1
        )  # 将动作索引归一化到[0, 1]范围内

        return norm_a

    def value_iteration_step(self):
        """执行一步值迭代"""
        pass

    def policy_evaluation(self):
        """评估当前策略，更新状态价值"""
        pass

    def policy_improvement(self):
        """策略改进：对每个状态选择最优动作"""
        pass

    @abstractmethod
    def train(self):
        pass

    def draw_policy(self, x, y, img):
        center_x, center_y = y * self.grid_size + 50, x * self.grid_size + 50
        for action_idx, action in enumerate(self.action_space):
            prob = self.policy[x, y, action_idx]
            length = prob * 40  # 根据概率调整箭头长度
            if action == [0, 0]:  # stay动作特殊处理
                if prob > 0.1:  # 如果stay动作的概率较大，画一个圆
                    cv2.circle(
                        img, (center_x, center_y), int(length // 2), (255, 0, 0), -1
                    )
            elif length > 0:  # 只有当概率大于0时才画箭头
                angle = np.arctan2(action[0], action[1])  # 计算箭头方向
                end_x = int(length * np.cos(angle)) + center_x
                end_y = int(length * np.sin(angle)) + center_y
                # end_x = length * action[1] * np.cos(45) + center_x
                # end_y = length * action[0] * np.cos(45) + center_y
                cv2.arrowedLine(
                    img,
                    (center_x, center_y),
                    (int(end_x), int(end_y)),
                    (0, 0, 0),
                    1,
                    tipLength=0.3,
                )

    def draw_state_value(self, x, y, img):
        text_value = self.state_values[x, y]
        cv2.putText(
            img,
            f"{text_value:.2f}",
            (y * self.grid_size + 10, x * self.grid_size + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    def draw_picture(self, wait_time=0):
        img = self.img.copy()
        for x in range(self.rows):
            for y in range(self.cols):
                self.draw_policy(x, y, img)
                self.draw_state_value(x, y, img)

        cv2.imshow("Grid World", img)
        cv2.waitKey(wait_time)

    def print_optimal_policy(self):
        # 提取最右侧略对应的动作符号
        self.optimal_policy = self.best_policy_str()

        print("Current grid layout:")
        print(self.grid)

        print("\nOptimal state values:")
        # 格式化输出，保留3位小数
        for i in range(self.rows):
            row_str = "  ".join(
                [f"{self.state_values[i, j]:7.3f}" for j in range(self.cols)]
            )
            print(row_str)

        print("\nOptimal policy:")
        for i in range(self.rows):
            row_str = "  ".join(
                [f"{self.optimal_policy[i, j]:4}" for j in range(self.cols)]
            )
            print(row_str)

    def agent_step(self, x, y):
        """执行一步智能体操作，调试专用"""
        img = self.img.copy()
        start_x, start_y = (
            y * self.grid_size + self.agent_size // 2,
            x * self.grid_size + self.agent_size // 2,
        )
        end_x, end_y = start_x + self.agent_size, start_y + self.agent_size
        img[
            start_x:end_x,
            start_y:end_y,
        ] = self.agent_img
        cv2.imshow("agent", img)
        cv2.waitKey(1)
