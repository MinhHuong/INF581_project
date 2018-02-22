import numpy as np
import random
from agent import Agent, Action
from matplotlib.pyplot import *


### ENVIRONMENT ###

class Environment:
    '''The class that represents our pretty environment'''

    # some defaults parameters
    default_parameters = {
        'width': 10,
        'height': 10,
        'obstacles': [(2, 2), (2, 3), (2, 4), (6, 7), (7, 7)],
        'nb_trashes': 30
    }

    def __init__(self, agent, w=0, h=0, nb_trashes=0):
        '''
        Initialize the environment

        :param agent: the agent to add in the environment
        :param w: width of the environment (not including walls)
        :param h: height of the environment (not including walls)
        :param nb_trashes: number of trashes in the environment
        '''

        self.width = self.default_parameters['width'] if w == 0 else w  # setting width
        self.height = self.default_parameters['height'] if h == 0 else h  # setting height

        self.obstacles = self.default_parameters['obstacles']  # set the obstacles

        # stuffs related to the Agent (action space, state space, the agent itself)
        self.action_space_n = Action.size()  # cardinality of action space
        self.state_space_n = (self.width + 1) * (self.height + 1)  # cardinality of action space : Position of agent
        self.agent = agent  # add the agent to the environment

        # start throwing trashes around to get the agent a job
        self.nb_trashes = self.default_parameters['nb_trashes'] if nb_trashes == 0 else nb_trashes
        self.trashes = []  # all positions of trashes
        i = 0
        random.seed(self.nb_trashes)  # to ensure that every time the random will return the same sequence
        while i < self.nb_trashes:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)

            # if newly generated position is not that of another trash / an obstacle / the initial position of the agent
            if (x, y) not in self.trashes and (x, y) not in self.obstacles and (x, y) != agent.position:
                self.trashes.append((x, y))
                i += 1

        # for conversion between position and tile #
        # this will help when using Q_table #
        self.pairs = np.array([(i, j) for i in range(self.width + 1) for j in range(self.height + 1)])
        self.fig = figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.xticks = np.arange(-0.5, self.width + 0.5, 1)
        self.yticks = np.arange(-0.5, self.height + 0.5, 1)
        self.ax.grid()
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.plot(np.array(self.trashes)[:, 0], np.array(self.trashes)[:, 1], "co", markersize=30, alpha=0.2)
        self.ax.plot(np.array(self.obstacles)[:, 0], np.array(self.obstacles)[:, 1], "ks", markersize=30, alpha=0.4)

    def display(self):
        '''
        Display the environment
        '''

        ion()
        self.ax.plot(self.agent.position[0], self.agent.position[1], "rX", markersize=30)
        show()
        pause(0.3)
        self.ax.plot(self.agent.position[0], self.agent.position[1], "ws", markersize=30)

    def go_into_obstacle(self, new_pos):
        '''
        Verify whether the agent hits an obstacle

        :param new_pos: next state after execution an action
        :return: True if new position is an obstacle
        '''

        # TO DISCUSS
        # we update new position disregard agent hitting an obstacle or not ? Why don't we do the same in case of walls ?
        self.agent.position = new_pos
        # TO DISCUSS

        return new_pos in self.obstacles

    def step(self, a):
        '''
        Execute action a

        :param a: an action in the action space {LEFT, RIGHT, UP, DOWN}
        :return: new state, reward, termination flag, info
        '''

        # prepare to calculate the new state
        go_into_wall = False
        go_into_obstacle = False
        new_pos = self.agent.position

        # TO DISCUSS
        # so, what is the limit of walls? -1 and width + 1 | height + 1 ?
        # TO DISCUSS

        # LEFT
        if a == Action.LEFT:
            if self.agent.position[0] == 0:
                go_into_wall = True
            else:
                new_pos = (self.agent.position[0] - 1, self.agent.position[1])
                go_into_obstacle = self.go_into_obstacle(new_pos)
        # RIGHT
        elif a == Action.RIGHT:
            if self.agent.position[0] == self.width:
                go_into_wall = True
            else:
                new_pos = (self.agent.position[0] + 1, self.agent.position[1])
                go_into_obstacle = self.go_into_obstacle(new_pos)
        # DOWN
        elif a == Action.DOWN:
            if self.agent.position[1] == 0:
                go_into_wall = True
            else:
                new_pos = (self.agent.position[0], self.agent.position[1] - 1)
                go_into_obstacle = self.go_into_obstacle(new_pos)
        # UP
        else:
            if self.agent.position[1] == self.height:
                go_into_wall = True
            else:
                new_pos = (self.agent.position[0], self.agent.position[1] + 1)
                go_into_obstacle = self.go_into_obstacle(new_pos)

        new_pos = self.pos2tile(new_pos)

        # default values
        reward = -1
        done = False
        info = "Cleaning brrr--- All is well!"

        # TO DISCUSS
        # stops both when hitting an obstacle / a wall ? (a wall is also a kind of obstacle)
        # TO DISCUSS

        # if the agent hits a wall or an obstacle, diminish the reward
        if go_into_wall or go_into_obstacle:
            reward = -2

        # if the agent hits an obstacle, the episode is done (because the agent is damaged and dead)
        if go_into_obstacle:
            done = True

        # if the agent manages to clean the trashes which is its job
        if self.agent.position in self.trashes:
            self.trashes.remove(self.agent.position)
            reward = 1
            info = "Find trash! Clean!"

        # if the environment is totally clean of trashes, my job here is done
        if len(self.trashes) == 0:
            done = True

        # hits walls and screams for help
        if go_into_wall:
            info = "Go into walls!"

        # gits obstacle and also screams for help
        if go_into_obstacle:
            info = "Go into obstacle!"

        return [new_pos, reward, done, info]

    def reset(self):
        '''
        Reinitialize the starting position, and put back the trashes that have been cleaned (right at the same position as previously initialized)

        Returns
        -------
        new_pos: new initial position
        '''

        self.trashes.clear()
        i = 0
        random.seed(self.nb_trashes)  # return the same sequence of random numbers
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
        self.agent.position = new_pos

        # drawing stuffs
        cla()
        self.ax.grid()
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.plot(np.array(self.trashes)[:, 0], np.array(self.trashes)[:, 1], "co", markersize=30, alpha=0.2)
        self.ax.plot(np.array(self.obstacles)[:, 0], np.array(self.obstacles)[:, 1], "ks", markersize=30, alpha=0.4)

        # return the new initial position
        return self.pos2tile(new_pos)

    def action_sample(self):
        '''
        Generate random action

        :return: action to execute
        '''
        return np.random.randint(0, self.action_space_n)

    def tile2pos(self, i):
        '''
        Flattened position to tuple

        :param i: tile number
        :return: position coordinates
        '''
        if i < 0:
            return None
        return self.pairs[int(i)]

    def pos2tile(self, pos):
        '''
        Tuple coordinate to flattened position

        :param pos: position ccordinates
        :return: tile number
        '''
        i = pos[0] * (self.width + 1) + pos[1]
        if i < (self.width + 1) * (self.height + 1):
            return i
        return -1

    def rollout(self, n_iter, pi):
        '''
        Generate the data (state, action, reward) for n_iter iterations in advanced by following policy pi

        :param n_iter: number of iterations to anticipate
        :param pi: the policy to follow (matrix of probability of actions to take at each state)
        :return: a set of states, actions, and reward
        '''
        states = []
        actions = []
        rewards = []

        s = self.agent.position
        sprime = self.agent.position
        states.append(self.pos2tile(s))
        for i in range(n_iter):
            a = np.argmax(pi[sprime])  # get the action that maximizes the policy map
            actions.append(a)
            sprime, reward, done, info = self.step(a)
            states.append(sprime)
            rewards.append(reward)

            if done:
                break

        self.agent.position = s
        return [states, actions, rewards]



