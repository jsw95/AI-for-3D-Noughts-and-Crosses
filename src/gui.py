from tkinter import *
from player import *
from ttt import Game


class GameButtons:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.Button_1 = Button(frame, text="1", command=lambda: self.print_num(1))
        self.Button_2 = Button(frame, text="2", command=self.callback)
        self.Button_3 = Button(frame, text="3", command=self.printMessage)
        self.Button_4 = Button(frame, text="4", command=self.printMessage)
        self.Button_5 = Button(frame, text="5", command=self.printMessage)
        self.Button_6 = Button(frame, text="6", command=self.printMessage)
        self.Button_7 = Button(frame, text="7", command=self.printMessage)
        self.Button_8 = Button(frame, text="8", command=self.printMessage)
        self.Button_9 = Button(frame, text="9", command=self.printMessage)

        self.Button_1.grid(row=0)
        self.Button_2.grid(row=0, column=1)
        self.Button_3.grid(row=0, column=2)
        self.Button_4.grid(row=1)
        self.Button_5.grid(row=1, column=1)
        self.Button_6.grid(row=1, column=2)
        self.Button_7.grid(row=2)
        self.Button_8.grid(row=2, column=1)
        self.Button_9.grid(row=2, column=2)
        p1 = AgentRL(0.01, 0.9, 0.9, model="rl_model", training=False)
        p2 = HumanPlayer()

        p3 = RandomPlayer()

        game = Game(p1, p2)
        game.play(p1, p2, agent_first=0, e=0, print_data=True)



        self.clicked = []

    def callback(self):
        self.clicked.append(widget.cget("text"))


    def print_num(self, number):
        print(number)
        return number

    def play_game(self):
        game.play(p1, p2, agent_first=0, e=0, print_data=True)

    def printMessage(self):
        print("PrintButton")



root = Tk()
GameButtons(root)
root.mainloop()




#
# self.quitButton = Button(frame, text="Quit", command=frame.quit)
# self.quitButton.gri(side=LEFT)
# button_1 = Button(root, text="Nine")
# button_1.bind("<Button-1>", print_name)
# button_1.pack()

# label_1 = Label(root, text="Name")
# label_2 = Label(root, text="Password")
# entry_1 = Entry(root)
# entry_2 = Entry(root)
#
# label_1.grid(row=0, sticky=E)
# label_2.grid(row=1, sticky=E)
#
# entry_1.grid(row=0, column=1)
# entry_2.grid(row=1, column=1)
#
# c = Checkbutton(root, text="Check this button")
# c.grid(row=2, columnspan=2)



# topFrame = Frame(root)
# topFrame.pack()  # Anytime you want to display something you must pack it
# bottomFrame = Frame(root)
# bottomFrame.pack(side=BOTTOM)
#
# button1 = Button(topFrame, text="Press me!", fg="red")
# button2 = Button(topFrame, text="Button 2", fg="blue")
# button3 = Button(bottomFrame, text="Button 3", fg="green")
# button4 = Button(bottomFrame, text="Button 4", fg="purple")
#
# button1.pack(side=LEFT, fill=BOTH, expand=True)
# button2.pack(side=RIGHT, fill=BOTH, expand=True)
# button3.pack(side=LEFT, fill=BOTH, expand=True)
# button4.pack(side=RIGHT, fill=BOTH, expand=True)

#
# one = Label(root, text="one", bg="green", fg="white")
# one.pack(fill=BOTH, expand=True)
#
# two = Label(root, text="two", bg="blue", fg="white")
# two.pack(fill=X)
#
# three = Label(root, text="three", bg="yellow", fg="white")
# three.pack(side=LEFT, fill=Y)
