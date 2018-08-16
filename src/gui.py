from tkinter import *

root = Tk()

topFrame = Frame(root)
topFrame.pack()  # Anytime you want to display something you must pack it
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

button1 = Button(topFrame, text="Press me!", fg="red")
button2 = Button(topFrame, text="Button 2", fg="blue")
button3 = Button(bottomFrame, text="Button 3", fg="green")
button4 = Button(bottomFrame, text="Button 4", fg="purple")

button1.pack(side=LEFT, fill=BOTH, expand=True)
button2.pack(side=RIGHT, fill=BOTH, expand=True)
button3.pack(side=LEFT, fill=BOTH, expand=True)
button4.pack(side=RIGHT, fill=BOTH, expand=True)

root.mainloop()








#
# one = Label(root, text="one", bg="green", fg="white")
# one.pack(fill=BOTH, expand=True)
#
# two = Label(root, text="two", bg="blue", fg="white")
# two.pack(fill=X)
#
# three = Label(root, text="three", bg="yellow", fg="white")
# three.pack(side=LEFT, fill=Y)
