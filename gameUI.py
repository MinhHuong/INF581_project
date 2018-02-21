import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg

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

# cleaning rate for each episode
clean_rate = []
crashes = []

crash_count = 0

for i_epsisode in range(20):
    s = env.reset()
    a = act_with_epsilon_greedy(s, q_table)
    for t in range(200):
        s_prime, reward, done, info = env.step(a)

        a_prime = act_with_epsilon_greedy(s_prime, q_table)

        delta = sarsa_update(q_table, s, a, reward, s_prime, a_prime)

        e_table[s, a] = e_table[s, a] + 1

        for u in range(n_s):
            for b in range(n_a):
                q_table[u, b] = q_table[u, b] + alpha * delta * e_table[u, b]
            e_table[u] = gamma * lamb * e_table[u]

        # Transition to new state
        s = s_prime
        a = a_prime

        crash = 0
        if info == "Go into walls!" or info == "Go into obstacle!":
            crash = 1

        crash_count += crash

        if done:
            break

    clean_rate.append((env.nb_trashes - len(env.trashes)) / env.nb_trashes)
    crashes.append(crash_count)

pygame.init()

pygame.font.init()
myfont = pygame.font.SysFont('Courier', 20)
height_score = 50

FPS = 20 # frames per second setting
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
crash_count = 0
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

    # Since the agent will not change position if it crashes the walls or obstacle
    # we need to take into account our graphic views

    crash = 0
    if info == "Go into walls!" or info == "Go into obstacle!":
        crash = 1

    crash_count += crash

    for tick in np.arange(10, facteur + 10, 10):
        DISPLAYSURF.fill(BROWN)
        for trash_pos in env.trashes:
            DISPLAYSURF.blit(dust_Img, (trash_pos[0] * facteur, trash_pos[1] * facteur))
        for rock_pos in env.obstacles:
            DISPLAYSURF.blit(rock_Img, (rock_pos[0] * facteur, rock_pos[1] * facteur))

        if a == Action.LEFT:
            DISPLAYSURF.blit(agent_Img, (env.agent_state[0] * facteur + (1 - crash) * (facteur - tick), env.agent_state[1] * facteur))
        elif a == Action.RIGHT:
            DISPLAYSURF.blit(agent_Img, (env.agent_state[0] * facteur - (1 - crash) * (facteur - tick), env.agent_state[1] * facteur))
        elif a == Action.DOWN:
            DISPLAYSURF.blit(agent_Img, (env.agent_state[0] * facteur, env.agent_state[1] * facteur + (1 - crash) * (facteur - tick)))
        elif a == Action.UP:
            DISPLAYSURF.blit(agent_Img, (env.agent_state[0] * facteur, env.agent_state[1] * facteur - (1 - crash) * (facteur - tick)))

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

    # Transition to new state
    s = s_prime
    a = a_prime

    if done:
        break

clean_rate.append((env.nb_trashes - len(env.trashes)) / env.nb_trashes)
crashes.append(crash_count)


import pylab

fig = pylab.figure(figsize=[4, 4],
                   dpi = 70,)

ax = fig.gca()
ax.set_title('Cleaning rate for each epoch')
ax.set_ylabel('Rate')
ax.set_xlabel('Episode')
ax.plot(clean_rate)

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()

while True:
    DISPLAYSURF.fill(WHITE)
    scoresurface = myfont.render("Cleaning rate : " + clean_count, False, (0, 0, 0))
    DISPLAYSURF.blit(scoresurface, (50, 450))
    crash_surface = myfont.render("Crash count : " + str(crash_count), False, (0, 0, 0))
    DISPLAYSURF.blit(crash_surface, (50, 500))

    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")
    DISPLAYSURF.blit(surf, (75, 0))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(FPS)