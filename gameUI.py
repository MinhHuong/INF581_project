import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg

import pygame, sys
from pygame.locals import *
from environment import Environment
from agent import Agent, Action
from main import *
import numpy as np
from utils import Converter


######################################################################################
#      teaching the agent to clean trashes properly with reinforcement learning      #
######################################################################################

agent = Agent(pos=(0,0)) # create a new agent
env = Environment() # add the agent to the environment
convert = Converter(env)

facteur = 50
agent_pos = env.position # get the agent's position
agent_pos = (agent_pos[0]*facteur, agent_pos[1]*facteur) # multiply it by a factor

n_a = env.action_space_n # get the action space size
n_s = env.state_space_n # get the state space size

q_table = np.zeros([n_s, n_a]) # init Q table
e_table = np.zeros([n_s, n_a]) # init eligibility traces

# cleaning rate for each episode
clean_rate = []
crashes = []
crash_count = 0
n_episodes = 1000
n_timesteps = 500

#q_table=agent.NFQ(env, n_timesteps)
for i_epsisode in range(n_episodes):
    s = convert.state2tile(env.reset())
    a = agent.get_action(s, q_table, method="greedy")

    for t in range(n_timesteps):
        s_prime, reward, done, info = env.step(a)
        s_prime = convert.state2tile(s_prime)

        a_prime = agent.get_action(s_prime, q_table, method="greedy")

        agent.update(q_table, s, a, reward, s_prime, a_prime, e_table)

        # Transition to new state
        s = s_prime
        a = a_prime

        #crash = 0
        #if info == "Go into walls!" or info == "Go into obstacle!":
        #    crash = 1

        #crash_count += crash

        if done:
            break

    clean_rate.append((env.nb_trashes - len(env.trashes)) / env.nb_trashes)
    crashes.append(crash_count)
    agent.epsilon = agent.epsilon * agent.epsilon_decay

##################################################
#      making real cute graphical interface      #
##################################################

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

path_img = 'img/'
agent_Img = pygame.image.load(path_img + 'Vicky_vac_copy.png')
rock_Img = pygame.image.load(path_img + 'rock_copy.png')
dust_Img = pygame.image.load(path_img + 'dust_copy.png')

q_table = np.loadtxt('q_table.dat')

clean_count = 0
crash_count = 0
s = convert.state2tile(env.reset())
nb_trashes = len(env.trashes)
a = agent.get_action(s, q_table, method="softmax")

# print out graphical interface for the LAST episode only !!!
for t in range(100):

    s_prime, reward, done, info = env.step(a)
    s_prime = convert.state2tile(s_prime)

    a_prime = agent.get_action(s_prime, q_table, method="softmax")

    agent.update(q_table, s, a, reward, s_prime, a_prime, e_table)

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
            DISPLAYSURF.blit(agent_Img, (env.position[0] * facteur + (1 - crash) * (facteur - tick), env.position[1] * facteur))
        elif a == Action.RIGHT:
            DISPLAYSURF.blit(agent_Img, (env.position[0] * facteur - (1 - crash) * (facteur - tick), env.position[1] * facteur))
        elif a == Action.DOWN:
            DISPLAYSURF.blit(agent_Img, (env.position[0] * facteur, env.position[1] * facteur + (1 - crash) * (facteur - tick)))
        elif a == Action.UP:
            DISPLAYSURF.blit(agent_Img, (env.position[0] * facteur, env.position[1] * facteur - (1 - crash) * (facteur - tick)))

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


####################################################################
#        plotting cleaning rate to verify agent's efficiency       #
####################################################################

import pylab

fig = pylab.figure(figsize=[4, 4],
                   dpi = 70,)

ax = fig.gca()
ax.set_title('Cleaning rate for each epoch')
ax.set_ylabel('Rate')
ax.set_xlabel('Episode')
ax.scatter(range(len(clean_rate)), clean_rate)

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()
clean_rate_txt = np.mean(clean_rate)

while True:
    DISPLAYSURF.fill(WHITE)
    scoresurface = myfont.render("Cleaning rate : " + str(clean_rate_txt), False, (0, 0, 0))
    DISPLAYSURF.blit(scoresurface, (50, 450))
    crash_surface = myfont.render("Crash count : " + str(crash_count), False, (0, 0, 0))
    DISPLAYSURF.blit(crash_surface, (50, 500))

    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")
    DISPLAYSURF.blit(surf, (120, 0))

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    fpsClock.tick(FPS)