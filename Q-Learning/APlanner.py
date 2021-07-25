import math
import numpy as np


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.position == other.position


def get_path(current_node, world):
    num_rows, num_cols = np.shape(world)
    path = []
    # set every location as 0
    result = [[0 for i in range(num_cols)] for j in range(num_rows)]
    current = current_node

    while current is not None:
        path.append(current.position)
        current = current.parent

    # invert path
    path = path[::-1]

    start_value = 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1

    return result

# A* planner
def plan(world, cost, start, end):
    start_node = Node(None, tuple(start))
    end_node = Node(None, tuple(end))
    start_node.g = start_node.h = start_node.f = 0
    end_node.g = end_node.h = end_node.f = 0

    not_visited = []
    visited = []
    not_visited.append(start_node)

    iterations = 0
    max_loops = (len(world) // 2) ** 10

    num_rows, num_cols = np.shape(world)

    move = [[-1, 0],
            [0, -1],
            [1, 0],
            [0, 1]]

    while len(not_visited) > 0:
        iterations += 1

        current_index = 0
        current_node = not_visited[0]
        for index, item in enumerate(not_visited):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        if iterations > max_loops:
            print("exceeded max iterations")
            return get_path(current_node, world)

        not_visited.pop(current_index)
        visited.append(current_node)

        # win condition
        if current_node == end_node:
            return get_path(current_node, world)

        children = []

        for new_position in move:

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (num_rows - 1) or node_position[0] < 0 or node_position[1] > (num_cols - 1) or node_position[1] < 0:
                continue

            if world[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:
            if len([visited_child for visited_child in visited if visited_child == child]) > 0:
                continue

            # Euclidean heuristic
            child.h = math.sqrt((((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)))
            child.g = current_node.g + cost
            child.f = child.g + child.h

            if len([i for i in not_visited if child == i and child.g > i.g]) > 0:
                continue

            not_visited.append(child)


if __name__ == '__main__':

    world = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]

    # start = [2, 0]
    start = [9, 6]
    end = [4, 5]
    cost = 1

    path = plan(world, cost, start, end)
    print('\n'.join([''.join(["{:" ">3d}".format(item) for item in row]) for row in path]))
