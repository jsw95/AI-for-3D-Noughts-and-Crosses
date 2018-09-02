from player import *
from ttt import *
import csv
import random
import matplotlib.pyplot as plt

with open("../data/2d-r-v-l-35000g.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = [i for i in reader]

# epoch_ax = [int(i) for i in data[0]]
epoch_ax = [i for i in range(0, 3500, 50)]
tab = [float(i) for i in data[-2]]
deep = [float(i) for i in data[2]]
# threeX200 = [float(i) for i in data[5]]
# twoX200 = [float(i) for i in data[3]]
# rewards_hist1 = [float(i) for i in data[3]]

plt.plot(epoch_ax, deep, label="Loss")
plt.plot(epoch_ax, tab, label="Reward")
# plt.plot(epoch_ax, threeX200, label="3x200")
ax = plt.subplot()
ax.axhline(y=0, color='k', linewidth=0.4)
# ax.set_yticks([0.0])
# ax.set_ylim(bottom=-1, top=1)
# ax.set_xlim(left=0, right=15001)

# ax.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], minor=True)
# # ax.set_yticks([])
# ax.yaxis.grid(True, which="major")
# ax.yaxis.grid(True, which="minor")

# plt.plot(epoch_ax, twoX200, label="2x200")
plt.legend()
# plt.grid(b=True)#, which="major", axis="y")
plt.xlabel("Epochs")
plt.ylabel("Average Reward/Loss")
plt.savefig("../figures/2d-r-v-l-35000g.pdf")
plt.show()














#
# def train_ai(
#         infile="3d-r-r-gl<25.csv",
#         outfile="3d",
#         hidden_units=10,
#         hidden_layers=5,
#         epochs=10,
#         batch_size=100,
#         learning_rate=0.25,
#         shape=64):
#
#     tf.reset_default_graph()
#
#     model_name = "{}-{}-{}-{}-{}-{}".format(
#             outfile, hidden_units, hidden_layers, epochs, batch_size, learning_rate)
#
#     with open("../data/" + infile, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         win_data = [i for i in reader if (i[-1] == '1') and (int(i[-2]) < 25)][:120]
#
#     inputs, labels, game_lengths = [], [], []
#     for row in win_data:
#         inputs.append([int(d) for d in row[:shape]])
#         labels.append([int(d) for d in row[shape][1:len(row[shape]) - 1].split(',')])
#         game_lengths.append(int(row[-2]))
#
#     print(game_lengths[100], labels[100], inputs[100])
#
#
# train_ai()


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

