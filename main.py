from environment import Environment
from agent import Agent
from utils import Converter
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def rolling_mean(array, n = 201):
    ret = np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main(explore, exploit, trace):
    agent = Agent(pos=(0, 0))               # create an agent at initial position (0,0)
    env = Environment()                     # create an environment
    convert = Converter(env)

    n_a = env.action_space_n                # get the size of action space
    n_s = env.state_space_n                 # get the size of state space

    q_table = np.zeros([n_s, n_a])

    if trace == 2:
        e_table = np.zeros([n_s, n_a])          # initialize the eligibility trace
    else:
        e_table = None
    if explore == 1:
        ex_method = "greedy"
    else:
        ex_method = "softmax"

    n_episode = 1000
    n_timestep = 500

    window = 200
    cleaning_rate = []
    returns = deque(maxlen=window)
    avg_rt_count = []


    # for each episode
    for i_episode in range(n_episode):
        s = convert.state2tile(env.reset())
        a = agent.get_action(s, q_table, method=ex_method)

        # for each epoch
        clean_rate = 0
        for t in range(n_timestep):

            # Act: take a step and receive (new state, reward, termination flag, additional information)
            s_prime, reward, done, info = env.step(a)
            agent.store_transition(s, a, reward)

            # if it is the last episode, print out info (to avoid print out too much)
            if (i_episode == n_episode - 1):
                env.display()
                print(info)

            # Select an action
            '''We need to give method explicitely {"softmax", "greedy"}'''
            good_acts = []
            (_x, _y) = (s_prime[0], s_prime[1])
            if (_x - 1, _y) in env.trashes:
                good_acts.append(0)
            if (_x + 1, _y) in env.trashes:
                good_acts.append(1)
            if (_x, _y - 1) in env.trashes:
                good_acts.append(2)
            if (_x, _y + 1) in env.trashes:
                good_acts.append(3)
            s_prime = convert.state2tile(s_prime)
            a_prime = agent.get_action(s_prime, q_table, good_acts=good_acts, method=ex_method)

            # Update a Q value table
            '''
            Update method is implicitely given according to
            the number of parameters
            '''
            if exploit == 1:
                agent.update(q_table, s, a, reward, s_prime, a_prime = None, e_table=e_table)
            else:
                agent.update(q_table, s, a, reward, s_prime, a_prime, e_table=e_table)

            # Transition to new state
            s = s_prime
            a = a_prime

            if done:
                reward_0 = agent.discounted_return()[0]
                clean_rate = (env.nb_trashes - len(env.trashes)) / env.nb_trashes
                returns.append(reward_0)
                avg_rt_count.append(np.average(returns))
                print("Episode: {0}\t Nb_Steps{1:>4}\t Epsilon: {2:.3f}\t Tau: {3:.3f}\t Clean Rate: {4:.3f}\t Discounted_return: {5:.3f}\t".format(
                    i_episode, t + 1, agent.epsilon, agent.tau, clean_rate, reward_0))
                # print(info)
                break

        agent.ep_rs.clear()
        agent.ep_obs.clear()
        agent.ep_as.clear()

        agent.epsilon = agent.epsilon * agent.epsilon_decay
        agent.tau = agent.init_tau + i_episode * agent.tau_inc
        cleaning_rate.append(clean_rate)

    plt.ioff()
    fig = plt.figure(figsize=(7, 9))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.scatter(range(n_episode), cleaning_rate, color='r', label="Cleaning rate")
    ax1.legend()
    ax2 = fig.add_subplot(3, 1, 2)
    moving_avg = rolling_mean(cleaning_rate, n = window)
    ax2.plot(range(len(moving_avg)), moving_avg, color='r', label="Rolling average cleaning rate")
    ax2.legend()
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(range(len(avg_rt_count)), avg_rt_count, color='r', label="Rolling average discounted return")
    ax3.legend()
    plt.show()

if __name__ == '__main__':
    explore = input("Exploration : (1 or 2) (greedy or softmax) ")
    if explore != "2":
        print("Greedy exploration")
        explore = 1
    else:
        print("Softmax selection")
        explore = int(explore)
    exploit = input("Exploitation : (1 or 2) (Q learning or Sarsa) ")
    if exploit != "2":
        print("Q-learning control")
        exploit = 1
    else:
        print("Sarsa control")
        exploit = int(exploit)
    trace = input("Eligibility trace : (1 or 2) (No or Yes) ")
    if trace != "2":
        print("Do not trace!")
        trace = 1
    else:
        print("Eligibility trace")
        trace = int(trace)
    main(explore, exploit, trace)