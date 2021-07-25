import numpy as np


class Q_Learner:
    def __init__(self, world, epsilon=0.4, alpha=0.1, gamma=1):
        self.world = world
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.q_matrix = dict()
        for x in range(world.height):
            for y in range(world.width):
                self.q_matrix[(x, y)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

    def choose_direction(self, possible_directions):
        if np.random.uniform(0, 1) < self.epsilon:
            chosen_direction = possible_directions[np.random.randint(0, len(possible_directions))]
        else:
            q_vals = self.q_matrix[self.world.robotLoc]
            maxValue = max(q_vals.values())
            chosen_direction = np.random.choice([k for k, v in q_vals.items() if v == maxValue])

        return chosen_direction

    def learn(self, p_state, c_state, reward, direction):
        q_vals = self.q_matrix[c_state]
        maxValue = max(q_vals.values())
        current_q = self.q_matrix[p_state][direction]

        self.q_matrix[p_state][direction] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * maxValue)

    def show_table(self):
        print(self.q_matrix)
