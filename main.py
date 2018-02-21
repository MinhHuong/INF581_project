from environment import Environment
from agent import Agent
import random
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    agent = Agent(pos=(0,0)) # create an agent
    env = Environment(agent) # create an environment

    #env.display()

    n_a = env.action_space_n # get the size of action space
    n_s = env.state_space_n # get the size of state space

    q_table = np.zeros([n_s, n_a])
    e_table = np.zeros([n_s, n_a])

    # for each episode
    for i_episode in range(20):
        s = env.reset()
        print("episode {}".format(i_episode))
        a = agent.act_with_epsilon_greedy(s, q_table)
        #a = act_with_softmax(s, q_table)
        total_return = 0.0

        # for each epoch
        for t in range(300):

            # Act
            s_prime, reward, done, info = env.step(a)

            total_return += np.power(agent.gamma, t) * reward
            if (i_episode > 18): # print out info every 18 episode
                env.display()
                print(info)

            # Select an action
            #a_prime = act_with_softmax(s, q_table)
            a_prime = agent.act_with_epsilon_greedy(s_prime, q_table)

            # update a Q value table
            delta = agent.sarsa_update(q_table, s, a, reward, s_prime, a_prime)
            #q_table[s, a] = q_learning_update(q_table, s, a, reward, s_prime)
            e_table[s, a] = e_table[s, a] + 1

            # Update q_table and e_table
            for u in range(n_s):
                for b in range(n_a):
                    q_table[u, b] = q_table[u, b] + agent.alpha * delta * e_table[u, b]
                e_table[u] = agent.gamma * agent.lamb * e_table[u]
            e_table[s] = e_table[s] / agent.gamma / agent.lamb

            # Transition to new state
            s = s_prime
            a = a_prime

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print(info)
                break

        # add all these inside Agent ?
        agent.epsilon = agent.epsilon * agent.decay_factor
        agent.tau = agent.init_tau + i_episode * agent.tau_inc
        print("total return {}".format(total_return))
        print("percentage of cleaning {}".format((30 - len(env.trashes))/30))
        #print("epsilon {}".format(epsilon))

    #print(q_table)
    #env.display()
    np.savetxt('q_table.dat', q_table, fmt='%f')
