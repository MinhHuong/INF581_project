from environment import Environment
from agent import Agent
import random
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    agent = Agent(pos=(0,0)) # create an agent at initial position (0,0)
    env = Environment(agent) # create an environment

    #env.display()

    n_a = env.action_space_n # get the size of action space
    n_s = env.state_space_n # get the size of state space

    q_table = np.zeros([n_s, n_a]) # initialize the Q table
    e_table = np.zeros([n_s, n_a]) # initialize the eligibility trace

    #pi = (1.0 / n_a) * np.ones([n_s, n_a]) # initialize the policy map
    #[states, actions, rewards] = env.rollout(20, pi) # generate 20 (state, action, reward) following pi policy map

    n_episode = 1000
    n_timestep = 100

    cleaning_rate = []

    '''
    if we use NFQ, we should comment the update method below
    '''
    #q_table = agent.NFQ(env, n_timestep)

    # for each episode
    for i_episode in range(n_episode):
        s = env.reset()
        #print("====== Episode {}".format(i_episode))
        a = agent.get_action(s, q_table, method="greedy")

        # for each epoch
        for t in range(n_timestep):

            # Act: take a step and receive (new state, reward, termination flag, additional information)
            s_prime, reward, done, info = env.step(a)

            # if it is the last episode, print out info (to avoid print out too much)
            if (i_episode == n_episode-1):
                env.display()
                print(info)

            # Select an action
            '''We need to give method explicitely {"softmax", "greedy"}'''
            a_prime = agent.get_action(s_prime, q_table, method="greedy")

            # Update a Q value table
            '''
            Update method is implicitely given according to
            the number of parameters
            '''
            agent.update(q_table, s, a, reward, s_prime, a_prime, e_table)

            # Transition to new state
            s = s_prime
            a = a_prime

            if done:
                clean_rate = (env.nb_trashes - len(env.trashes))/env.nb_trashes
                print("Episode: {0}\t Nb_Steps{1:>4}\t Epsilon: {2:.3f}\t Clean Rate: {3:.3f}\t".format(i_episode, t + 1, agent.epsilon, clean_rate))
                #print(info)
                break

        # add all these inside Agent ?
        agent.epsilon = agent.epsilon * agent.epsilon_decay
        agent.tau = agent.init_tau + i_episode * agent.tau_inc

        #print("total return {}".format(total_return))
        cleaning_rate.append(clean_rate)

    #print(q_table)
    #env.display()
    #np.savetxt('q_table.dat', q_table, fmt='%f')
    print(np.mean(cleaning_rate))
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(n_episode), cleaning_rate, label = "Cleaning rate")
    ax.legend()
    plt.show()
