# from ttt import Game
from ttt_3D import Game
import player
import csv

p1 = player.RandomPlayer()
p2 = player.RandomPlayer()
p3 = player.HumanPlayer()
# p4 = player.AIPlayer(model="model")
p6 = player.AIPlayer(model="model_deep")
# p5 = player.AIPlayer(model="model_deep_10")
# p7 = player.AIPlayer(model="model_lr0.1")
# p8 = player.AIPlayer(model="model_50_lr0.1_100e") # not working due to shape error?
# p9 = player.AIPlayer(model="model_20_30") # not working due to shape error?
p9 = player.AIPlayer(model="model_20_50v2") # not working due to shape error?


def play_game(player1, player2, write_data=False, outfile="databoy.csv", iter=1):

    p1_wins, p2_wins, draws = 0, 0, 0

    for i in range(iter):
        # if i % 10 == 0:
        #     print("Epoch: {}".format(i))

        game = Game(player1, player2)

        if player1.human or player2.human:
            game.print = True
            game.print_board()

        while not game.end:
            if not game.end:
                game.advance(player1.move(game.board))
            if not game.end:
                game.advance(player2.move(game.board))

        if game.winner == 1:
            p1_wins += 1
        elif game.winner == -1:
            p2_wins += 1
        elif game.winner == 0:
            draws += 1

        if write_data:
            with open("../data/" + outfile, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(game.output_data())

    print("P1 - D - P2\n{} - {} - {}".format(p1_wins, draws, p2_wins))


# play_game(p1, p9, iter=500, write_data=True)
# play_game(p9, p1, iter=100, write_data=True)

# play_game(p9, p6, iter=25)
# play_game(p6, p9, iter=25)
play_game(p1, p2, write_data=True, outfile="3d-r-r-1000.csv", iter=10000)


