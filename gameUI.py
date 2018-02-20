import pygame, sys
from pygame.locals import *
from environment import Environment
from main import *
import numpy as np

env = Environment()

facteur = 50

agent_pos = env.agent_state

agent_pos = (agent_pos[0]*facteur, agent_pos[1]*facteur)

n_a = env.action_space_n
n_s = env.state_space_n

q_table = np.zeros([n_s, n_a])
e_table = np.zeros([n_s, n_a])

for i_epsisode in range(20):
    s = env.reset()
    a = act_with_epsilon_greedy(s, q_table)
    for t in range(100):
        s_prime, reward, done, info = env.step(a)

        a_prime = act_with_epsilon_greedy(s_prime, q_table)

        delta = sarsa_update(q_table, s, a, reward, s_prime, a_prime)

        e_table[s, a] = e_table[s, a] + 1

        for u in range(n_s):
            for b in range(n_a):
                q_table[u, b] = q_table[u, b] + alpha * delta * e_table[u, b]
            e_table[u] = gamma * lamb * e_table[u]
        e_table[s] = e_table[s] / gamma / lamb

        # Transition to new state
        s = s_prime
        a = a_prime


pygame.init()

pygame.font.init()
myfont = pygame.font.SysFont('Courier', 20)
height_score = 50

FPS = 5 # frames per second setting
fpsClock = pygame.time.Clock()


# set up the window
WIDTH = 550
HEIGHT = 550
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT + height_score), 0, 32)
pygame.display.set_caption('Animation')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BROWN = (245,245,220)
agent_Img = pygame.image.load('Vicky_vac_copy.png')
rock_Img = pygame.image.load('rock_copy.png')
dust_Img = pygame.image.load('dust_copy.png')

clean_count = 0
s = env.reset()
nb_trashes = len(env.trashes)
a = act_with_epsilon_greedy(s,q_table)

for t in range(100):

    s_prime, reward, done, info = env.step(a)

    a_prime = act_with_epsilon_greedy(s_prime, q_table)

    delta = sarsa_update(q_table, s, a, reward, s_prime, a_prime)

    e_table[s, a] = e_table[s, a] + 1

    for u in range(n_s):
        for b in range(n_a):
            q_table[u, b] = q_table[u, b] + alpha * delta * e_table[u, b]
        e_table[u] = gamma * lamb * e_table[u]
    e_table[s] = e_table[s] / gamma / lamb

    # Transition to new state
    s = s_prime
    a = a_prime

    DISPLAYSURF.fill(BROWN)
    for trash_pos in env.trashes:
        DISPLAYSURF.blit(dust_Img, (trash_pos[0]*facteur, trash_pos[1]*facteur))
    for rock_pos in env.obstacles:
        DISPLAYSURF.blit(rock_Img, (rock_pos[0]*facteur, rock_pos[1]*facteur))
    DISPLAYSURF.blit(agent_Img, (env.agent_state[0]*facteur,env.agent_state[1]*facteur))
    pygame.draw.line(DISPLAYSURF, RED, (0, HEIGHT), (WIDTH, HEIGHT), 2)

    clean_count = str(nb_trashes - len(env.trashes)) + "/" + str(nb_trashes)
    scoresurface = myfont.render(clean_count, False, (0, 0, 0))
    infosurface = myfont.render(info, False, RED)
    DISPLAYSURF.blit(scoresurface, (20, HEIGHT + 13))
    DISPLAYSURF.blit(infosurface, (200, HEIGHT + 13))


    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(FPS)