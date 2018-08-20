import numpy as np


class Game(object):

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.player_current = 1
        self.winner = 0
        self.board = [0] * 64
        self.print = False
        self.winners = self.generate_wins()
        self.total_reward = 0
        self.agent_wins = 0
        self.random_wins = 0
        self.draws = 0

    @staticmethod
    def generate_wins():
        wins = []

        for i in range(16):
            wins.append([j + i * 4 for j in range(4)])  # horizontal wins
            wins.append([i + j for j in [0, 16, 32, 48]])  # depth wins

        for i in range(4):
            k = i * 16
            p = i * 4
            wins.extend([[k + j, k + j + 4, k + j + 8, k + j + 12] for j in range(4)])  # vertical wins
            # all diagonal face wins
            wins.extend([[k + 0, k + 5, k + 10, k + 15]])
            wins.extend([[k + 12, k + 9, k + 6, k + 3]])
            wins.extend([[p, p + 17, p + 34, p + 51]])
            wins.extend([[p + 3, p + 18, p + 33, p + 48]])
            wins.extend([[i, i + 20, i + 40, i + 60]])
            wins.extend([[i + 12, i + 24, i + 36, i + 48]])

        # all vertex diagonal wins
        wins.extend([
            [0, 21, 42, 63],
            [48, 37, 26, 15],
            [3, 22, 41, 60],
            [12, 25, 38, 51]
        ])

        return wins

    def reset_game(self):
        self.board = [0] * 64
        self.player_current = 1
        self.winner = None

    def win_check(self):
        for win in self.winners:
            # print(win)
            if (self.board[win[0]] == self.board[win[1]]
                    and self.board[win[0]] == self.board[win[2]]
                    and self.board[win[0]] == self.board[win[3]]
                    and self.board[win[0]] != 0):
                self.winner = self.player_current
                self.end = True
                return True

    def draw_check(self):
        if 0 not in self.board:
            self.end = True
            return True  # returns true for draw but no winner




    #
    # def move(self, pos):
    #     if pos < 0 or pos > 63:
    #         print("Please enter a square between 1 and 64")
    #         # self.move(pos=pos)
    #
    #     elif self.board[pos] != 0:
    #         print("Please choose a blank square")
    #         # self.move(pos=pos)
    #     else:
    #         self.board[pos] = self.player_current

    # def print_board(self):
    #     def f(l):
    #         if l == 0:
    #             return ' '
    #         elif l == 1:
    #             return 'X'
    #         elif l == -1:
    #             return 'O'
    #
    #     b = self.board  # Cleaner formatting
    #
    #     for i in range(4):
    #         print('\n-----------------')
    #         print('| {} | {} | {} | {} |'.format(
    #             f(self.board[0 + 16 * i]), f(self.board[1 + 16 * i]), f(self.board[2 + 16 * i]),
    #             f(self.board[3 + 16 * i])))
    #         print('-----------------')
    #         print('| {} | {} | {} | {} |'.format(
    #             f(self.board[4 + 16 * i]), f(self.board[5 + 16 * i]), f(self.board[6 + 16 * i]),
    #             f(self.board[7 + 16 * i])))
    #         print('-----------------  Layer {}'.format(i + 1))
    #
    #         print('| {} | {} | {} | {} |'.format(
    #             f(self.board[8 + 16 * i]), f(self.board[9 + 16 * i]), f(self.board[10 + 16 * i]),
    #             f(self.board[11 + 16 * i])))
    #         print('-----------------')
    #         print('| {} | {} | {} | {} |'.format(
    #             f(self.board[12 + 16 * i]), f(self.board[13 + 16 * i]), f(self.board[14 + 16 * i]),
    #             f(self.board[15 + 16 * i])))
    #         print('\\---------------\\')



    def print_board(self):
        def f(l):
            if l == 0:
                return ' '
            elif l == 1:
                return 'X'
            elif l == -1:
                return 'O'

        b = self.board
        print("\n******************************************************************************\n")
        print("     Layer 1             Layer 2             Layer 3             Layer 4        ")
        print('-----------------   ' * 4)
        print('| {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |'.format(
            f(b[0]), f(b[1]), f(b[2]), f(b[3]), f(b[16]), f(b[17]), f(b[18]), f(b[19]),
            f(b[32]), f(b[33]), f(b[34]), f(b[35]), f(b[48]), f(b[49]), f(b[50]), f(b[51])))
        print('-----------------   ' * 4)
        print('| {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |'.format(
            f(b[4]), f(b[5]), f(b[6]), f(b[7]), f(b[20]), f(b[21]), f(b[22]), f(b[23]),
            f(b[36]), f(b[37]), f(b[38]), f(b[39]), f(b[52]), f(b[53]), f(b[54]), f(b[55])))
        print('-----------------   ' * 4)
        print('| {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |'.format(
            f(b[8]), f(b[9]), f(b[10]), f(b[11]), f(b[24]), f(b[25]), f(b[26]), f(b[27]),
            f(b[40]), f(b[41]), f(b[42]), f(b[43]), f(b[56]), f(b[57]), f(b[58]), f(b[59])))
        print('-----------------   ' * 4)
        print('| {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |   | {} | {} | {} | {} |'.format(
            f(b[12]), f(b[13]), f(b[14]), f(b[15]), f(b[28]), f(b[29]), f(b[30]), f(b[31]),
            f(b[44]), f(b[45]), f(b[46]), f(b[47]), f(b[60]), f(b[61]), f(b[62]), f(b[63])))
        print('-----------------   ' * 4)
        print("\n")


    def free_squares(self):
        free = [i for i in range(len(self.board)) if self.board[i] == 0]
        return free

    def play(self, agent, player, sess=None, prediction=None, q_vals=None, inputs=None,
             agent_first=1, e=0, print_data=False):
        """
        Main game playing function. Each call is one game.
        Can choose random or human player against an agent.
        Agent Q-table is updated after each game based on reward.
        Returns game history
        """

        self.reset_game()

        # Storing history of moves and boards (only relevant ones)
        board_log, board_prev_log, move_log = [], [], []
        # n_moves = 0

        if agent_first < 0.5:  # Who moves first
            if player.player == "agentRL":
                move_ = player.move([i * self.player_current for i in self.board], e=player.epsilon)
            else:
                move_ = player.move(self.board)
            self.board[int(move_)] = self.player_current  # update board with move
            self.player_current *= -1
            if print_data:
                self.print_board()

        while True:  # Training loop
            if print_data:
                self.print_board()

            board_prev = [i * self.player_current for i in self.board]

            if agent.training is False:
                move = agent.move(board_prev, e)  # Agents move

            else:
                move = agent.training_move(board_prev, e, sess, prediction, q_vals, inputs)  # Agents move

            self.board[int(move)] = self.player_current

            board = [i * self.player_current for i in self.board]

            move_log.append(move)
            board_prev_log.append(board_prev)
            board_log.append(board)

            if self.win_check():
                reward = 1
                self.agent_wins += 1
                # self.total_reward += reward
                break

            if self.draw_check():
                reward = 0
                self.draws += 1
                break

            self.player_current *= -1
            if print_data:
                self.print_board()

            if player.player == "agentRL":
                move_ = player.move(board, e=player.epsilon)
            else:
                move_ = player.move(self.board)
            self.board[move_] = self.player_current

            if self.win_check():
                reward = -5
                self.random_wins += 1
                # self.total_reward += reward
                break

            if self.draw_check():
                reward = 0
                self.draws += 1
                break

            self.player_current *= -1

        # agent.update_qtable(board_log, board_prev_log, move_log, reward)
        reward_log = [0] * len(move_log)
        reward_log[-1] = reward

        return [board_prev_log, move_log, board_log, reward_log]




    # def output_data(self):
    #
    #     def ohe(i):
    #         enc = [0] * 64
    #         enc[i] = 1
    #         return enc
    #
    #     data_log = []
    #     for i in range(len(self.move_log)):
    #         data = []
    #         data.extend(self.board_log[i])
    #         data.append(ohe(self.move_log[i][0])) # one-hot moves
    #         data.append(len(self.move_log)) # Length of game
    #         data.append(self.winner * self.move_log[i][1]) # whether player won
    #         data_log.append(data)
        # return data_log

    # def advance(self, pos):
    #
    #     self.board_log.append([i * self.player_current for i in self.board])
    #     self.move_log.append([pos, self.player_current])
    #
    #     self.move(pos=pos)
    #
    #     if self.print:
    #         self.print_board()
    #
    #     self.win_check()
    #     self.output_data()
    #     self.player_current *= -1





