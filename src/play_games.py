# from ttt import Game
from ttt_3D import Game
import player
import csv
from train import train_ai


def play_game(player1, player2, write_data=False, outfile="databoy.csv", iter=1, print_board=False):

    p1_wins, p2_wins, draws = 0, 0, 0

    for i in range(iter):
        if i % 25 == 0:
            print("Games: {}".format(i))

        game = Game(player1, player2)

        if print_board:
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

    if str(player1.player()) == "AIPlayer":
        with open("../models/info/" + player1.model + ".txt", 'a+') as info:
            info.write("\n{} games as first player vs {}:\n".format(iter, str(player2.player())))
            info.write("W - D - L:\n")
            info.write("{} - {} - {}:\n".format(p1_wins, draws, p2_wins))
            info.close()

    elif str(player2.player()) == "AIPlayer":
        with open("../models/info/" + player2.model + ".txt", 'a+') as info:
            info.write("\n{} games as second player vs {}:\n".format(iter, str(player1.player())))
            info.write("W - D - L:\n")
            info.write("{} - {} - {}:\n".format(p2_wins, draws, p1_wins))
            info.close()

    print("P1 - D - P2\n{} - {} - {}".format(p1_wins, draws, p2_wins))


if __name__ == "__main__":

    p1 = player.RandomPlayer()
    p2 = player.HumanPlayer()

    # train_ai(infile="3d-r-r-gl<25.csv",
    #          hidden_units=64,
    #          hidden_layers=1,
    #          epochs=10,
    #          batch_size=15,
    #          learning_rate=0.9)

    p3 = player.AIPlayer(model="3d-64-1-10-15-0.5")
    play_game(p3, p1, iter=5, write_data=True, outfile="3d-t-r-gl-1000.csv")
    play_game(p1, p3, iter=5, write_data=True, outfile="3d-t-r-gl-1000.csv")

    # train_ai(infile="3d-r-r-gl<25.csv",
    #          hidden_units=64,
    #          hidden_layers=1,
    #          epochs=10,
    #          batch_size=15,
    #          learning_rate=0.05)

    p4 = player.AIPlayer(model="3d-100-1-10-15-0.5")
    play_game(p4, p1, iter=5, write_data=True, outfile="3d-t-r-gl-1000.csv")
    play_game(p1, p4, iter=5, write_data=True, outfile="3d-t-r-gl-1000.csv")
