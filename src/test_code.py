from itertools import groupby
import csv
import numpy as np
def train_ai(
        infile="3d-t-r-gl<25.csv",
        hidden_units=10,
        hidden_layers=5,
        epochs=20,
        batch_size=100,
        learning_rate=0.25,
        outfile="3d-{}-{}-{}-{}-{}.txt".format(hidden_units, hidden_layers, epochs, batch_size, learning_rate),
        infofile=outfile,
        shape=64):
    print(shape)


import scipy.stats as sc





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

