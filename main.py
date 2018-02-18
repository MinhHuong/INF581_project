from environment import Environment
from enum import IntEnum
import random
import numpy as np

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # I am still concerned abt this action version #

env = Environment()

"""
env.display()
obs, r, flag, info = env.step(Action.UP)
print(info)
print(env.tile2pos(env.pos2tile(obs)))
env.display()
obs, r, flag, info = env.step(Action.DOWN)
print(info)
print(env.pos2tile(obs))
env.display()
obs, r, flag, info = env.step(Action.DOWN)
print(info)
print(env.pos2tile(obs))
env.display()
obs, r, flag, info = env.step(Action.RIGHT)
print(info)
print(env.pos2tile(obs))
env.display()
obs, r, flag, info = env.step(Action.RIGHT)
print(info)
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
decay_factor = 0.999
epsilon = 0.5
gamma = 0.99

def act_with_epsilon_greedy(s, q):
    '''
    take an action according to the epsilon_greedy policy

    :param s: the state of the environment
    :param q: the q_table for action value function
    :param env: the environment (only for position conversion)
    :return:
    '''
    a = np.argmax(q[s, :])
    if np.random.rand() < epsilon:
        a = env.action_sample()
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
    :param env: the environment (only for position conversion)
    :return: the update score
    '''
    td = r + gamma * q[s_prime, a_prime] - q[s, a]
    return q[s, a] + alpha * td

q_table = np.zeros([n_s, n_a])


for i_episode in range(20):
    s = env.reset()
    print("episode {}".format(i_episode))
    #env.display()
    a = act_with_epsilon_greedy(env.pos2tile(s), q_table)
    for t in range(100):
        #print("at timestep {}".format(t))
        s_prime, reward, done, info = env.step(a)
        #env.display()

        a_prime = act_with_epsilon_greedy(env.pos2tile(s_prime), q_table)

        # update a Q value table
        q_table[env.pos2tile(s), a] = sarsa_update(q_table, env.pos2tile(s), a, reward, env.pos2tile(s_prime), a_prime)

        s = s_prime
        a = a_prime

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    print("percentage of cleaning {}".format(len(env.trashes)/30))
    epsilon = epsilon * decay_factor
