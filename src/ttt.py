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

        def f(pl):
            if pl == 1:
                return self.player1
            elif pl == -1:
                return self.player2

        for win in self.winners:
            if (self.board[win[0]] == self.board[win[1]]
                    and self.board[win[0]] == self.board[win[2]]
                    and self.board[win[0]] != 0):

                self.winner = self.player_current
                self.end = True

                # print("{} wins!".format(f(self.player_current)))

            # else:
            #
            #     print("no")
            #     return False

    def move(self, pos):

        if pos < 0 or pos > 8:
            print("Please enter a square between 0 and 8")
            self.move(pos=pos)

        elif self.board[pos] != 0:
            print("Please choose a blank square")

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

    def advance(self, pos):
        # print("-" * 40)
        self.board_log.append([i for i in self.board])
        self.move_log.append([pos, self.player_current])
        self.move(pos=pos)
        self.win_check()
        self.output_data()
        # self.print_board()
        self.player_current *= -1

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

# game = Game('Jack', 'Bob')


# while not game.end:
#
#     if len(game.free_squares()) > 0:
#         move = np.random.choice(game.free_squares())
#         game.advance(move)
#     else:
#         break
#
#
# [print(i) for i in game.output_data()]
# print(game.output_data())

