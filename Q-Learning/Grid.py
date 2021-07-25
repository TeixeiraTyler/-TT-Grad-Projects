import numpy as np
import matplotlib.pyplot as plt


ROWS = 10
COLUMNS = 10
W_POS = (4, 5)
S_POS = [(0, 0), (0, 9), (9, 0), (9, 9)]  # possible starting locations
MAX_STEPS = 150
BLOCKERS = [(2, 1), (3, 1), (6, 5), (9, 3), (7, 6)]


class GridWorld:
    def __init__(self):
        self.height = ROWS
        self.width = COLUMNS
        self.winLoc = W_POS
        self.finished = False
        self.num_steps = 0
        self.grid = np.zeros((self.height, self.width)) - 1
        self.heat = np.zeros((self.height, self.width)) - 1
        self.grid[self.winLoc[0], self.winLoc[1]] = 100
        self.directions = ['up', 'down', 'left', 'right']

        choice = np.random.randint(0, 12)
        if choice < 3:
            self.robotLoc = S_POS[0]
            self.start = S_POS[0]
        elif choice < 6:
            self.robotLoc = S_POS[1]
            self.start = S_POS[1]
        elif choice < 9:
            self.robotLoc = S_POS[2]
            self.start = S_POS[2]
        else:
            self.robotLoc = S_POS[3]
            self.start = S_POS[3]

    # returns the value at the specified position.
    def getReward(self, position):
        return self.grid[position[0], position[1]]

    # moves the robot in the direction specified and returns the value of the new position.
    def make_step(self, direction):
        previous_location = self.robotLoc
        skip = False

        if direction == 'up':
            next_location = (self.robotLoc[0] - 1, self.robotLoc[1])
            if previous_location[0] == 0:                   # at top
                reward = self.getReward(previous_location)
            else:
                for blk in BLOCKERS:
                    if next_location == blk:
                        reward = self.getReward(previous_location)
                        skip = True
                if not skip:
                    self.robotLoc = next_location
                    reward = self.getReward(self.robotLoc)

        elif direction == 'down':
            next_location = (self.robotLoc[0] + 1, self.robotLoc[1])
            if previous_location[0] == self.height - 1:     # at bottom
                reward = self.getReward(previous_location)
            else:
                for blk in BLOCKERS:
                    if next_location == blk:
                        reward = self.getReward(previous_location)
                        skip = True
                if not skip:
                    self.robotLoc = next_location
                    reward = self.getReward(self.robotLoc)

        elif direction == 'left':
            next_location = (self.robotLoc[0], self.robotLoc[1] - 1)
            if previous_location[1] == 0:                   # at left-most
                reward = self.getReward(previous_location)
            else:
                for blk in BLOCKERS:
                    if next_location == blk:
                        reward = self.getReward(previous_location)
                        skip = True
                if not skip:
                    self.robotLoc = next_location
                    reward = self.getReward(self.robotLoc)

        else:
            next_location = (self.robotLoc[0], self.robotLoc[1] + 1)
            if previous_location[1] == self.width - 1:      # at right-most
                reward = self.getReward(previous_location)
            else:
                for blk in BLOCKERS:
                    if next_location == blk:
                        reward = self.getReward(previous_location)
                        skip = True
                if not skip:
                    self.robotLoc = next_location
                    reward = self.getReward(self.robotLoc)

        self.heat[self.robotLoc[0], self.robotLoc[1]] += 10

        if self.robotLoc == W_POS or self.num_steps >= MAX_STEPS:
            self.finished = True

        self.num_steps += 1
        return reward

    def show_world(self):
        print(np.matrix(self.grid))

    def show_heatmap(self):
        plt.imshow(self.heat, cmap='hot', interpolation='nearest')
        plt.show()
