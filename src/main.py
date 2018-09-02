import numpy as np
# from ttt import Game
from ttt_3D import Game
from player import *
import random
import matplotlib.pyplot as plt
import csv
import pickle


if __name__ == "__main__":

    p1x = AgentRL(0, model="3d-working-50k-winner", training=False)
    # p1 = AgentRL(0, model="2d-layers-nodes-2x200", training=False)
    # p1x = AgentRL(0, model="test3", training=False)

    p1 = AgentRLTable(0.01, 0.9, 0.9, model="qtable.pkl")
    # p2 = AgentRLTable(0.01, 0.9, 0.9)

    pS = SmartPlayer()
    p4 = RandomPlayer()
    p3 = HumanPlayer()
    reward_log = []
    game = Game(p1, pS)



    iter = 100
    for i in range(iter):
        if i % 10 == 0:
            print("\nGame number: {}".format(i))
            reward_log.append((game.agent_wins - game.random_wins)/500)
            print((game.agent_wins - game.random_wins)/1000)
            game.random_wins = 0
            game.agent_wins = 0
        # game.play(p1, pS, agent_first=i%2, e=max(0.1, (iter - i)/iter), print_data=False)
        game.play(p1x, pS, agent_first=i % 2, e=0, print_data=False)



