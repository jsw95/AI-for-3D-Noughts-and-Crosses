import numpy as np
# from ttt import game
from ttt_3D import Game
from player import RandomPlayer
from player import HumanPlayer
from player import AgentRL
import random
import matplotlib.pyplot as plt
import tensorflow as tf




# Training and testing -- move to new file
if __name__ == "__main__":

    p1 = AgentRL(0, model="baseline_3d", training=False)
    # p1x = AgentRL(0.01, 0.9, 0.9, model="rl_model2", training=False)
    p2 = RandomPlayer()
    p3 = HumanPlayer()
    reward_log = []
    game = Game(p1, p2)
    n = random.random()



    iter = 5
    for i in range(iter):
        n = random.random()
        if i % 100 == 0:
            print("\nGame number: {}".format(i))
        game.play(p1, p3, agent_first=n, e=0, print_data=True)
        reward_log.append(game.total_reward)




    print("Agent wins: {}".format(game.agent_wins))
    print("Draws: {}".format(game.draws))
    print("Random wins: {}".format(game.random_wins))
    # print(p1.Qtable["[0, 0, 0, 0, 0, 0, 0, 0, 0]"])