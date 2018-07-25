import numpy as np


class Game(object):

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.player_current = 1
        self.winner = 0
        self.end = False
        # self.board = [[0] * 9 for i in range(3)]
        self.board = [0] * 9
        self.board_log = []
        self.move_log = []
        self.print = False

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

    """ fix this shit """
    def move(self, pos):
        if pos < 0 or pos > 8:
            print("Please enter a square between 0 and 8")
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

        print('\n-------------')
        print('| {} | {} | {} |'.format(f(self.board[0]), f(self.board[1]), f(self.board[2])))
        print('-------------')
        print('| {} | {} | {} |'.format(f(self.board[3]), f(self.board[4]), f(self.board[5])))
        print('-------------')
        print('| {} | {} | {} |'.format(f(self.board[6]), f(self.board[7]), f(self.board[8])))
        print('-------------\n')

    def output_data(self):

        def ohe(i):
            enc = [0] * 9
            enc[i] = 1
            return enc

        data_log = []
        for i in range(len(self.move_log)):
            data = []
            data.extend(self.board_log[i])
            data.append(ohe(self.move_log[i][0]))
            data.append(self.winner * self.move_log[i][1])
            data_log.append(data)

        return data_log

    def free_squares(self):
        free = [i for i in range(9) if self.board[i] == 0]
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

