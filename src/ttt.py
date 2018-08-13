import numpy as np
from player import RandomPlayer
from trainRL import AgentRL
import random
import matplotlib.pyplot as plt

class Game(object):

    def __init__(self, agent, player):
        self.agent = agent
        self.player = player
        self.player_current = 1
        self.winner = None
        self.end = False
        self.board = [0] * 9
        self.board_log = []
        self.move_log = []
        self.print = False
        self.total_reward = 0
        self.agent_wins = 0
        self.random_wins = 0


    winners = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]

    def win_check(self):
        for win in self.winners:
            if (self.board[win[0]] == self.board[win[1]]
                    and self.board[win[0]] == self.board[win[2]]
                    and self.board[win[0]] != 0):

                self.winner = self.player_current
                self.end = True
                return True

        if 0 not in self.board:
            self.end = True
            return True  # returns true for draw but no winner


    # def move(self, pos):
    #     if pos < 0 or pos > 8:
    #         print("Please enter a square between 0 and 8")
    #         # self.move(pos=pos)
    #
    #     elif self.board[pos] != 0:
    #         print("Please choose a blank square")
    #         # self.move(pos=pos)
    #     else:
    #         self.board[pos] = self.player_current


    def reset_game(self):
        self.board = [0] * 9
        self.player_current = 1
        self.winner = None
        self.end = False


    def print_board(self):
        def f(l):
            if l == 0:
                return ' '
            elif l == 1:
                return 'X'
            elif l == -1:
                return 'O'

        print('\n-------------')
        print('| {} | {} | {} |'.format(f(self.board[0]), f(self.board[1]), f(self.board[2])))
        print('-------------')
        print('| {} | {} | {} |'.format(f(self.board[3]), f(self.board[4]), f(self.board[5])))
        print('-------------')
        print('| {} | {} | {} |'.format(f(self.board[6]), f(self.board[7]), f(self.board[8])))
        print('-------------\n')

    def free_squares(self):
        free = [i for i in range(9) if self.board[i] == 0]
        return free


    def play(self, agent, player, training=True, agent_first=1):

        if agent_first < 0.5:
            print("Computer moves first")
            move = player.move(self.board)
            self.board[int(move)] = self.player_current  # update board with move
            self.player_current *= -1

        # print("Starting while loop")
        while True:
            board_prev = str(self.board)
            move = agent.move(self.board)

            # print("Computer move: {}".format(move))
            self.board[int(move)] = self.player_current

            win = self.win_check()
            if win:
                # print("Win? {}".format(win))
                reward = 1
                self.agent_wins += 1
                print("Agent Player Wins")


                self.total_reward += reward

                agent.update_qtable(str(self.board), board_prev, str(move), reward)

                break

            self.player_current *= -1

            Rmove = player.move(self.board)
            # print("Random move: {}".format(Rmove))
            self.board[Rmove] = self.player_current

            win = self.win_check()


            if win:
                # self.print_board()
                # print("Win? {}".format(win))
                reward = -1
                self.random_wins += 1
                print("Random Player Wins")
                self.total_reward += reward
                break

            else:
                reward = 0



            self.player_current *= -1
            agent.update_qtable(str(self.board), board_prev, str(move), reward)
            board_prev = self.board


        self.reset_game()










p1 = AgentRL(0.1, 0.9, 0, 1)
p2 = RandomPlayer()
# print(p1.player)
reward_log = []
game = Game(p1, p2)
for i in range(25000):
    n = random.random()
    print("\nGame number: {}".format(i))
    game.play(p1,p2, agent_first=n)
    reward_log.append(game.total_reward)
print("Agent wins: {}".format(game.agent_wins))
print("Random wins: {}".format(game.random_wins))
# p1.print_Qtable()
plt.plot(reward_log)
plt.show()






    #        self.board_log.append([i * self.player_current for i in self.board])
    #     self.move_log.append([pos, self.player_current])
    # def output_data(self):
    #
    #     def ohe(i):
    #         enc = [0] * 9
    #         enc[i] = 1
    #         return enc
    #
    #     data_log = []
    #     for i in range(len(self.move_log)):
    #         data = []
    #         data.extend(self.board_log[i])
    #         data.append(ohe(self.move_log[i][0]))
    #         data.append(self.winner * self.move_log[i][1])
    #         data_log.append(data)
    #
    #     return data_log