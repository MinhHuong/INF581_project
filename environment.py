import numpy as np
import random
from matplotlib.pyplot import *

### ENVIRONMENT ###

class Environment:
    '''The class that represents our pretty environment'''
    default_parameters = {
        'width': 10,
        'height': 10,
        'obstacles': [(2, 2), (2, 3), (2, 4), (6, 7), (7, 7)],
        'nb_trashes': 30
    }

    def __init__(self, pos = (0, 0), w=0, h=0, nb_trashes=0):
        '''
        class initialisation

        :param w: width of the environment (not including walls)
        :param h: height of the environment (not including walls)
        :param nb_trashes: number of trashes in the environment
        '''

        self.width = self.default_parameters['width'] if w == 0 else w

        self.height = self.default_parameters['height'] if h == 0 else h

        self.nb_trashes = self.default_parameters['nb_trashes'] if nb_trashes == 0 else nb_trashes

        self.obstacles = self.default_parameters['obstacles']

        self.action_space_n = 4 # cardinality of action space : {LEFT, RIGHT, UP, DOWN} = {0, 1, 2, 3}

        self.state_space_n = (self.width+1) * (self.height+1) # cardinality of action space : Position of agent

        self.agent_state = pos

        # randomize positions of trashes
        self.trashes = []
        i = 0
        random.seed(self.nb_trashes)
        while i < self.nb_trashes:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if (x, y) not in self.trashes and (x, y) not in self.obstacles:
                self.trashes.append((x, y))
                i += 1
        # what if the starting point coincides with the trashes position #

        # for conversion between position and tile #
        # this will help when using Q_table #
        self.pairs = np.array([(i, j) for i in range(self.width+1) for j in range(self.height+1)])

    def display(self, ax):
        '''
        display the environment

        :return:None
        '''
        """
        for i in range(self.height + 1):
            for j in range(self.width + 1):
                if j < self.width:
                    if (i, j) == self.agent_state:
                        symbol = 'o'
                    elif (i, j) in self.obstacles:
                        symbol = '#'
                    elif (i, j) in self.trashes:
                        symbol = '*'
                    else:
                        symbol = ' '
                    print('| %s ' % symbol, end='',
                          flush=True)  # don't bother these parameters I only use those to print on same line
                else:
                    print('|', end='', flush=True)
            print()
        """
        ion()
        ax.plot(self.agent_state[0], self.agent_state[1], "rs", markersize=30)
        show()
        pause(0.3)
        ax.plot(self.agent_state[0], self.agent_state[1], "ws", markersize=30)


    def go_into_obstacle(self, new_pos):
        '''

        :param new_pos: next state after execution an action
        :return: True if new position is an obstacle
        '''
        if new_pos in self.obstacles:
            return True
        else:
            self.agent_state = new_pos
            return False

    def step(self, a):
        '''
        execute action a

        :param a: an action in {LEFT, RIGHT, UP, DOWN}
        :return: new state, reward, termination flag, info
        '''

        # calculate new state
        go_into_wall = False
        go_into_obstacle = False
        new_pos = self.agent_state
        if a == 0:                                                      # LEFT
            if self.agent_state[0] == 0:
                go_into_wall = True
            else:
                new_pos = (self.agent_state[0] - 1, self.agent_state[1])
                go_into_obstacle = self.go_into_obstacle(new_pos)
        elif a == 1:                                                    # RIGHT
            if self.agent_state[0] == self.width - 1:
                go_into_wall = True
            else:
                new_pos = (self.agent_state[0] + 1, self.agent_state[1])
                go_into_obstacle = self.go_into_obstacle(new_pos)
        elif a == 2:                                                    # DOWN
            if self.agent_state[1] == 0:
                go_into_wall = True
            else:
                new_pos = (self.agent_state[0], self.agent_state[1] - 1)
                go_into_obstacle = self.go_into_obstacle(new_pos)
        else:                                                           # UP
            if self.agent_state[1] == self.height - 1:
                go_into_wall = True
            else:
                new_pos = (self.agent_state[0], self.agent_state[1] + 1)
                go_into_obstacle = self.go_into_obstacle(new_pos)

        # calculate reward
        reward = -1
        done = False
        if go_into_wall or go_into_obstacle:
            reward = -5
            #done = True
        if self.agent_state in self.trashes:
            self.trashes.remove(self.agent_state)
            reward = 0
        if len(self.trashes) == 0:
            done = True
        info = "All is well!"
        if go_into_wall:
            info = "Go into walls!"
        if go_into_obstacle:
            info = "Go into obstacle!"
        return [new_pos, reward, done, info]

    def reset(self):
        '''
        reinitialize the starting position, and put back the trashes cleaned

        :return:
        '''
        self.trashes.clear()
        i = 0
        random.seed(self.nb_trashes)
        while i < self.nb_trashes:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if (x, y) not in self.trashes and (x, y) not in self.obstacles:
                self.trashes.append((x, y))
                i += 1
        # random position starting point for robot
        # new position must be different from obstacles!
        new_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        while new_pos in self.obstacles:
            new_pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
        self.agent_state = new_pos
        return new_pos

    def action_sample(self):
        '''
        generate random action

        :return: action to execute
        '''
        return np.random.randint(0, self.action_space_n)

    def tile2pos(self, i):
        '''
        :param i: tile number
        :return: position coordinates
        '''
        if i < 0:
            return None
        return self.pairs[int(i)]

    def pos2tile(self, pos):
        '''
        :param pos: position ccordinates
        :return: tile number
        '''
        i = pos[0] * (self.width+1) + pos[1]
        if i < (self.width+1) * (self.height+1):
            return i
        return -1
