import numpy as np


class Game(object):

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.player_current = 1
        self.winner = 0
        self.end = False
        self.board = [0] * 64
        self.board_log = []
        self.move_log = []
        self.print = False
        self.winners = self.generate_wins()

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

        if 0 not in self.board:
            self.end = True



    def move(self, pos):
        if pos < 0 or pos > 63:
            print("Please enter a square between 1 and 64")
            # self.move(pos=pos)

        elif self.board[pos] != 0:
            print("Please choose a blank square")
            # self.move(pos=pos)
        else:
            self.board[pos] = self.player_current

    def print_board(self):
        def f(l):
            if l == 0:
                return ' '
            elif l == 1:
                return 'X'
            elif l == -1:
                return 'O'

        for i in range(4):
            print('\n-----------------')
            print('| {} | {} | {} | {} |'.format(
                f(self.board[0 + 16 * i]), f(self.board[1 + 16 * i]), f(self.board[2 + 16 * i]),
                f(self.board[3 + 16 * i])))
            print('-----------------')
            print('| {} | {} | {} | {} |'.format(
                f(self.board[4 + 16 * i]), f(self.board[5 + 16 * i]), f(self.board[6 + 16 * i]),
                f(self.board[7 + 16 * i])))
            print('-----------------  Layer {}'.format(i + 1))

            print('| {} | {} | {} | {} |'.format(
                f(self.board[8 + 16 * i]), f(self.board[9 + 16 * i]), f(self.board[10 + 16 * i]),
                f(self.board[11 + 16 * i])))
            print('-----------------')
            print('| {} | {} | {} | {} |'.format(
                f(self.board[12 + 16 * i]), f(self.board[13 + 16 * i]), f(self.board[14 + 16 * i]),
                f(self.board[15 + 16 * i])))
            print('\\---------------\\')

    def output_data(self):

        def ohe(i):
            enc = [0] * 64
            enc[i] = 1
            return enc

        data_log = []
        for i in range(len(self.move_log)):
            data = []
            data.extend(self.board_log[i])
            data.append(ohe(self.move_log[i][0])) # one-hot moves
            data.append(len(self.move_log)) # Length of game
            data.append(self.winner * self.move_log[i][1]) # whether player won
            data_log.append(data)

        return data_log

    def free_squares(self):
        free = [i for i in range(64) if self.board[i] == 0]
        return free

    def advance(self, pos):

        self.board_log.append([i * self.player_current for i in self.board])
        self.move_log.append([pos, self.player_current])

        self.move(pos=pos)

        if self.print:
            self.print_board()

        self.win_check()
        self.output_data()
        self.player_current *= -1





