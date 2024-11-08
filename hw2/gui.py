import tkinter
import os
import train  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import read
filename = ""
def updateFileName(f):
    global file
    file.set(f)
    global filename
    filename = f
    print(filename)
def trainfun():
    X, y = read.readFile(filename)
    model = train.train(X, y, 0.1, 100)
with open('dataset/edge.txt') as f:
    startLine = f.readline().strip()
    startx, starty, startTheta = map(int,startLine.split(','))
    goalLine1 = f.readline().strip()
    goalx1, goaly1 = map(int,goalLine1.split(','))
    goalLine2 = f.readline().strip()
    goalx2, goaly2 = map(int,goalLine2.split(','))
    edgex = []
    edgey = []
    for line in f:
        x, y = map(int,line.split(','))
        edgex.append(x)
        edgey.append(y)
root = tkinter.Tk()
root.wm_title("Self Driving Car")
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

file_menu = tkinter.Menu(root)
menu = tkinter.Menu(file_menu)
menu.add_command(label="train4dAll", command=lambda f = "train4dAll":updateFileName(f))
menu.add_command(label="train6dAll", command=lambda f = "train6dAll":updateFileName(f))
file_menu.add_cascade(label="File", menu=menu)
root.config(menu=file_menu)

file = tkinter.StringVar()
file.set("No file selected")
fileLabel = tkinter.Label(root, textvariable=file, font=("Times",15,"bold"))
fileLabel.grid(row=0, column=0, columnspan=4)

fig, ax = plt.subplots()
ax.plot(startx, starty, 'bo', markersize=10)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((goalx1, goaly1), abs(goalx1 - goalx2), abs(goaly1 - goaly2), fill=None, edgecolor='black'))
ax.plot(edgex, edgey)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky='nsew')

frame = tkinter.Frame(root)
frame.grid(row=2, column=0, columnspan=4)
trainButton = tkinter.Button(frame, text="Train", command=trainfun)
trainButton.grid(row=2, column=0, columnspan=2)

runButton = tkinter.Button(frame, text="Run", command=trainfun)
runButton.grid(row=2, column=2, columnspan=2)

tkinter.mainloop()