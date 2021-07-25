from Grid import GridWorld
import matplotlib.pyplot as plt

from Q_Learner import Q_Learner


def run(world, learner, episodes, learn, test):
    reward_per_episode = []

    for episode in range(episodes):
        sum_reward = 0
        step = 0
        finished = False
        while not finished:
            p_state = world.robotLoc
            direction = learner.choose_direction(world.directions)
            reward = world.make_step(direction)
            c_state = world.robotLoc

            if learn == True:
                learner.learn(p_state, c_state, reward, direction)

            sum_reward += reward
            step += 1

            if world.finished:
                temp = world.heat
                world.__init__()
                world.heat = temp
                finished = True

        reward_per_episode.append(sum_reward)

    if test:
        plt.plot(reward_per_episode)
        plt.show()
        world.show_heatmap()

        world.__init__()
        test_r = run(world, learner, episodes=2, learn=False, test=False)
        plt.plot(test_r)
        plt.show()
        world.show_heatmap()

    return reward_per_episode  # Return performance log


world = GridWorld()
learner = Q_Learner(world)

run(world, learner, episodes=500, learn=True, test=True)

learner.show_table()

print("Finished")
