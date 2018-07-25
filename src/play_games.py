from ttt import Game
import player
import csv

p1 = player.RandomPlayer()
p2 = player.RandomPlayer()
p3 = player.HumanPlayer()
p4 = player.AIPlayer()
p5 = player.AIPlayer()


def play_game(player1, player2, write_data=False):

    p1_wins, p2_wins, draws = 0, 0, 0

    for i in range(20):

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
            with open("../data/data-30000.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(game.output_data())

    print("Player 1 wins - Draws - Player 2 wins \n{} - {} - {}".format(p1_wins, draws, p2_wins))


play_game(p2, p5)



