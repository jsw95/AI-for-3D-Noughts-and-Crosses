from player import *
from ttt import *
import csv
import random
import numpy as np


def train_ai(
        infile="3d-r-r-gl<25.csv",
        outfile="3d",
        hidden_units=10,
        hidden_layers=5,
        epochs=10,
        batch_size=100,
        learning_rate=0.25,
        shape=64):

    tf.reset_default_graph()

    model_name = "{}-{}-{}-{}-{}-{}".format(
            outfile, hidden_units, hidden_layers, epochs, batch_size, learning_rate)

    with open("../data/" + infile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        win_data = [i for i in reader if (i[-1] == '1') and (int(i[-2]) < 25)][:120]

    inputs, labels, game_lengths = [], [], []
    for row in win_data:
        inputs.append([int(d) for d in row[:shape]])
        labels.append([int(d) for d in row[shape][1:len(row[shape]) - 1].split(',')])
        game_lengths.append(int(row[-2]))

    print(game_lengths[100], labels[100], inputs[100])


train_ai()


# with open("../data/3d-r-r-gl-1000.csv", newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     win_data = [i for i in reader if (i[-1] == '1') and (int(i[-2]) < 25)]
#
# print("Read")
#
# with open("../data/3d-r-r-gl<25.csv", 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(win_data)

# lens = [int(i[0]) for i in groupby(data)]
#
# print(lens)
# print(np.mean(lens))
# print(np.min(lens))
# print(sc.describe(lens))
#
# print(len([i for i in lens if i <= 20]))

