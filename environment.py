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
        'obstacles': [(2, 2), (3, 2), (4, 2), (7, 1), (8, 1), (8,2),
                      (8, 3), (8, 4), (4, 3), (4, 4), (4, 5), (5, 5),
                      (5, 6), (5, 7), (1, 7), (1, 8), (2, 8), (3, 8), (6, 7), (7, 7)],
        'nb_trashes': 22
    }

    def __init__(self, w=0, h=0, nb_trashes=0):
        '''
        Initialize the environment

        :param agent: the agent to add in the environment
        :param w: width of the environment (not including walls)
        :param h: height of the environment (not including walls)
        :param nb_trashes: number of trashes in the environment
        '''

        self.width = self.default_parameters['width'] if w == 0 else w      # setting width
        self.height = self.default_parameters['height'] if h == 0 else h    # setting height

        self.obstacles = self.default_parameters['obstacles']       # set the obstacles

        # stuffs related to the Agent (action space, state space, the agent itself)
        self.action_space_n = Action.size()                         # cardinality of action space
        self.state_space_n = 2 * (self.width + 2) * (self.height + 2)   # cardinality of action space : Position of agent
        self.state_features = 3                                         # position and presence of reward
        self.position = (0, 0)

        # start throwing trashes around to get the agent a job
        self.nb_trashes = self.default_parameters['nb_trashes'] if nb_trashes == 0 else nb_trashes
        self.clean_trash = 0
        self.trashes = []  # all positions of trashes
        i = 0
        random.seed(self.nb_trashes)  # to ensure that every time the random will return the same sequence
        while i < self.nb_trashes:
            x = random.randint(1, self.width+1)
            y = random.randint(1, self.height+1)

            # if newly generated position is not that of another trash / an obstacle / the initial position of the agent
            if (x, y) not in self.trashes and (x, y) not in self.obstacles and (x, y) != self.position:
                self.trashes.append((x, y))
                i += 1

        # for conversion between position and tile #
        # this will help when using Q_table #
        self.fig = figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.xticks = np.arange(-0.5, self.width + 1.5, 1)
        self.yticks = np.arange(-0.5, self.height + 1.5, 1)
        self.ax.grid()
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.plot(np.array(self.trashes)[:, 0], np.array(self.trashes)[:, 1], "co", markersize=30, alpha=0.2)
        self.ax.plot(np.array(self.obstacles)[:, 0], np.array(self.obstacles)[:, 1], "ks", markersize=30, alpha=0.4)
        # drawing the walls
        self.ax.plot(range(self.width + 2), 0 * np.ones(self.width + 2), 'r')
        self.ax.plot(range(self.width + 2), (self.height + 1) * np.ones(self.width + 2), 'r')
        self.ax.plot(0 * np.ones(self.height + 2), range(self.height + 2), 'r')
        self.ax.plot((self.width + 1) * np.ones(self.height + 2), range(self.height + 2), 'r')

    def display(self):
        '''
        Display the environment
        '''

        ion()
        self.ax.plot(self.position[0], self.position[1], "rX", markersize=30)
        show()
        pause(0.3)
        self.ax.plot(self.position[0], self.position[1], "ws", markersize=30)

    def go_into_obstacle(self, new_pos):
        '''
        Verify whether the agent hits an obstacle

        :param new_pos: next state after execution an action
        :return: True if new position is an obstacle
        '''

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

        # TO DISCUSS
        # so, what is the limit of walls? -1 and width + 1 | height + 1 ?
        # TO DISCUSS
        '''
        Rescale the map in order to add wall position 
        Wall position:
        [0, 0]          , [0, 1]         ,..., [0, HEIGHT + 2]          (Left wall)
        [WIDTH + 2, 0]  , [WIDTH + 2, 1] ,..., [WIDTH + 2, HEIGHT + 2]  (Right wall)
        [0, 0]          , [1, 0],        ,..., [WIDTH + 2, 0]           (Bottom wall)
        [0, HEIGHT + 2] , [1, HEIGHT + 2],..., [WIDTH + 2, HEIGHT + 2]  (Upper wall)
        After this, add terminal condition when robot goes into the wall
        '''

        # LEFT
        if a == Action.LEFT:
            new_pos = (self.position[0] - 1, self.position[1])
            if new_pos[0] == 0:
                go_into_wall = True
            else:
                go_into_obstacle = self.go_into_obstacle(new_pos)
        # RIGHT
        elif a == Action.RIGHT:
            new_pos = (self.position[0] + 1, self.position[1])
            if new_pos[0] == self.width + 1:
                go_into_wall = True
            else:
                go_into_obstacle = self.go_into_obstacle(new_pos)
        # DOWN
        elif a == Action.DOWN:
            new_pos = (self.position[0], self.position[1] - 1)
            if new_pos[1] == 0:
                go_into_wall = True
            else:
                go_into_obstacle = self.go_into_obstacle(new_pos)
        # UP
        else:
            new_pos = (self.position[0], self.position[1] + 1)
            if new_pos[1] == self.height + 1:
                go_into_wall = True
            else:
                go_into_obstacle = self.go_into_obstacle(new_pos)
        '''
        if go_into_wall:
            new_pos = self.agent.position   # to prevent the case where pos2tile return -1
        else:
            self.agent.position = new_pos
        '''
        self.position = new_pos
        #new_pos = self.pos2tile(new_pos)
        state = np.append(np.array(new_pos), 0)

        # default values
        reward = -0.01
        done = False
        info = "(" + str(self.position) + ", 0)"

        # if the agent hits a wall or an obstacle, diminish the reward
        if go_into_wall or go_into_obstacle:
            done = True
            reward = -1

        # if the agent manages to clean the trashes which is its job
        if self.position in self.trashes:
            self.trashes.remove(self.position)
            self.clean_trash += 1
            info = "(" + str(self.position) + ", 1)"
            reward = 1
            #new_pos += (self.height + 2) * (self.width + 2)  # the state (x, y, bool) where bool represent the presence of trash
            state[2] = 1


        # hits walls and screams for help
        if go_into_wall:
            info = "Go into walls!"

        # hits obstacle and also screams for help
        if go_into_obstacle:
            info = "Go into obstacle!"

        return [state, reward, done, info]

    def reset(self):
        '''
        Reinitialize the starting position, and put back the trashes that have been cleaned (right at the same position as previously initialized)

        Returns
        -------
        new_pos: new initial position
        '''

        self.trashes.clear()
        self.clean_trash = 0
        i = 0
        random.seed(self.nb_trashes)  # return the same sequence of random numbers
        while i < self.nb_trashes:
            x = random.randint(1, self.width)
            y = random.randint(1, self.height)
            if (x, y) not in self.trashes and (x, y) not in self.obstacles:
                self.trashes.append((x, y))
                i += 1

        # random position starting point for robot
        # new position must be different from obstacles!
        new_pos = (np.random.randint(1, self.width+1), np.random.randint(1, self.height+1))
        while new_pos in self.obstacles:
            new_pos = (np.random.randint(1, self.width+1), np.random.randint(1, self.height+1))
        self.position = new_pos
        state = np.append(np.array(new_pos), 0)

        # drawing stuffs
        cla()
        self.ax.grid()
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.plot(np.array(self.trashes)[:, 0], np.array(self.trashes)[:, 1], "co", markersize=25, alpha=0.2)
        self.ax.plot(np.array(self.obstacles)[:, 0], np.array(self.obstacles)[:, 1], "ks", markersize=25, alpha=0.4)
        # drawing the walls
        self.ax.plot(range(self.width+2), np.zeros(self.width+2), 'r')
        self.ax.plot(range(self.width+2), (self.height + 1) * np.ones(self.width+2), 'r')
        self.ax.plot(np.zeros(self.height+2), range(self.height+2), 'r')
        self.ax.plot((self.width + 1) * np.ones(self.height+2), range(self.height+2), 'r')

        # return the new initial position
        #new_pos = self.pos2tile(new_pos)
        if self.position in self.trashes:
            #new_pos = new_pos + (self.width + 2) * (self.height + 2)
            state[2] = 1
        return state

    def action_sample(self):
        '''
        Generate random action

        :return: action to execute
        '''
        return np.random.randint(0, self.action_space_n)

    # Not being used at the moment
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

        s = self.position
        trashes = self.trashes.copy()
        sprime = self.position
        states.append(self.pos2tile(s))
        for i in range(n_iter):
            a = np.argmax(pi[sprime])  # get the action that maximizes the policy map
            actions.append(a)
            sprime, reward, done, info = self.step(a)
            states.append(sprime)
            rewards.append(reward)

            if done:
                break

        # put back the intial state
        self.position = s
        self.trashes = trashes

        return [states, actions, rewards]