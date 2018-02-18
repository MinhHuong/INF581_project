from environment import Environment
from enum import IntEnum
import random
import numpy as np
import matplotlib.pyplot as plt

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3
    # I am still concerned abt this action version #

env = Environment()

#env.display(ax)
"""
obs, r, flag, info = env.step(Action.UP)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.DOWN)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.DOWN)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.RIGHT)
print(info)
print(r)
env.display()
obs, r, flag, info = env.step(Action.RIGHT)
print(info)
print(r)
env.display()
"""
n_a = env.action_space_n
n_s = env.state_space_n

#print("Nb_Actions: {}".format(n_a))
#print("Nb_States: {}".format(n_s))

'''
print("reset test")
env.reset()
env.display()

print("reset test")
env.reset()
env.display()
'''

# parameters for the RL agent
alpha = 0.1
decay_factor = 0.95
epsilon = 0.6
gamma = 0.99

def act_with_epsilon_greedy(s, q):
    '''
    take an action according to the epsilon_greedy policy

    :param s: the state of the environment
    :param q: the q_table for action value function
    :return:
    '''
    a = np.argmax(q[s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q.shape[1])
    #print(Action(a))
    return a

def sarsa_update(q, s, a, r, s_prime, a_prime):
    '''
    :param q: the q_table for action value function
    :param s: old state
    :param a: old action
    :param r: reward
    :param s_prime: new state
    :param a_prime: new action
    :return: the update score
    '''
    td = r + gamma * q[s_prime, a_prime] - q[s, a]
    return q[s, a] + alpha * td

def q_learning_update(q, s, a, r, s_prime):
    '''
    :param q: the q_table for action value function
    :param s: old state
    :param a: old action
    :param r: reward
    :param s_prime: new state
    :return: the update score
    '''
    td = r + gamma * np.max(q[s_prime, :]) - q[s, a]
    return q[s, a] + alpha * td


q_table = np.zeros([n_s, n_a])


fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.grid()
xticks = np.arange(-0.5, env.width + 0.5, 1)
yticks = np.arange(-0.5, env.height + 0.5, 1)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.plot(np.array(env.trashes)[:, 0], np.array(env.trashes)[:, 1], "co", markersize=30, alpha=0.2)
ax.plot(np.array(env.obstacles)[:, 0], np.array(env.obstacles)[:, 1], "ks", markersize=30, alpha=0.4)

for i_episode in range(20):
    s = env.reset()
    print("episode {}".format(i_episode))
    a = act_with_epsilon_greedy(env.pos2tile(s), q_table)
    total_return = 0.0
    for t in range(100):

        # Act
        s_prime, reward, done, info = env.step(a)

        total_return += np.power(gamma, t) * reward
        if (i_episode > 18):
            env.display(ax)

        # Select an action
        a_prime = act_with_epsilon_greedy(env.pos2tile(s_prime), q_table)

        # update a Q value table
        q_table[env.pos2tile(s), a] = sarsa_update(q_table, env.pos2tile(s), a, reward, env.pos2tile(s_prime), a_prime)
        #q_table[env.pos2tile(s), a] = q_learning_update(q_table, env.pos2tile(s), a, reward, env.pos2tile(s_prime))

        # Transition to new state
        s = s_prime
        a = a_prime

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print(info)
            break
    print("total return {}".format(total_return))
    print("percentage of cleaning {}".format((30 - len(env.trashes))/30))
    epsilon = epsilon * decay_factor
