"""
Initialization: 
    α_{t}(s, a) = α > 0 for all (s, a) and all t. ∈ (0, 1). 
    Initial q_{0}(s, a) for all (s, a). Initial ϵ-greedy policy π_{0} derived from q_{0}.
Goal: 
    Learn an optimal policy that can lead the agent to the target state from an initial state s_{0}.

For each episode, do
    Generate a_{0} at s_{0} following π_{0}(s_{0})
    If s_{t} (t = 0, 1, 2, . . . ) is not the target state, do
        Collect an experience sample (r_{t+1}, s_{t+1}, a_{t+1}) given (s_{t}, a_{t}):
            generate r_{t+1}, s_{t+1} by interacting with the environment; 
            generate a_{t+1} following π_{t}(s_{t+1}).

        Update q-value for (s_{t}, a_{t}):
            q_{t+1}(s_{t}, a_{t}) = q_{t}(s_{t}, a_{t}) - α_{t}(s_{t}, a_{t})  [  q_{t}(s_{t}, a_{t}) - (r_{t+1} + γq_{t}(s_{t+1}, a_{t+1}))  ]
        Update policy for s_{t}:
            π_{t+1}(a|s_{t}) = 1 - (ϵ/|A(s_{t})|) (|A(s_{t})| - 1) if a = argmax_{a} q_{t+1}(s_{t}, a)
            π_{t+1}(a|s_{t}) = ϵ/|A(s_{t})| otherwise
        s_{t+1} ← s_{t+1}, a_{t+1} ← a_{t+1}
"""

from GridWorld import GridWorldEnv, HyperParameters
import numpy as np
import tqdm


class Sarsa(GridWorldEnv):

    def __init__(self, HP: HyperParameters, action_space=9, newgrid=False):
        super().__init__(HP, action_space, newgrid)
        self.episode_length = 500
        self.lr = 0.001


    def value_iteration_step(self):
        x, y = np.random.randint(0, self.rows), np.random.randint(0, self.cols)
        a, ai = self.select_action(x, y)
        action_value_stable = True
        for _ in range(self.episode_length):
            r, (x2, y2) = self.get_next_state_and_reward((x, y), a)
            a2, ai2 = self.select_action(x2, y2)
            q_s_a_t = self.action_values[x, y, ai]
            q_s1_a1_t = self.action_values[x2, y2, ai2]
            # update q-value
            q_s_a_t1 = q_s_a_t - self.lr * (q_s_a_t - (r + self.gamma * q_s1_a1_t))
            if np.abs(q_s_a_t1 - self.action_values[x, y, ai]) > self.HP.end_condition:
                action_value_stable = False
            self.action_values[x, y, ai] = q_s_a_t1

            # update policy
            self.policy[x, y] = self.epsilon_greedy(
                np.argmax(self.action_values[x, y]), self.epsilon
            )
            self.state_values[x, y] = np.dot(
                self.action_values[x, y], self.policy[x, y]
            )
            self.draw_picture(1)
            x, y = x2, y2
            a, ai = a2, ai2
        return action_value_stable

    def train(self):
        iter = tqdm.tqdm(range(self.HP.max_iterations))
        for i in iter:
            policy_stable = self.value_iteration_step()
            if policy_stable:
                print(f"Policy converged after {i+1} iterations.")
                break

        self.print_optimal_policy()
        self.draw_picture()


hp = HyperParameters()
hp.max_iterations = 10000000
hp.gamma = 0.9
hp.rows = 5
hp.cols = 5
env = Sarsa(hp)
env.train()
