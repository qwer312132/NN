import tkinter
import train  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import read
import run
import numpy as np
import threading
from tkinter import ttk
filename = ""

def updateFileName(f):
    global file
    file.set(f)
    global filename
    filename = f
    print(filename)

def trainfun(update_progress, progress_bar):
    global model
    X, y = read.readFile(filename)
    model = train.train(X, y, 0.1, 1000, update_progress, progress_bar)

def update_progress(epoch, total_epochs, progress_bar):
    progress = (epoch / total_epochs) * 100
    progress_bar['value'] = progress
    root.update_idletasks()

def startTrain():
    train_thread = threading.Thread(target=trainfun, args=(update_progress,progress_bar))
    train_thread.start()

def runfun():
    path = run.run(startx, starty, startTheta, model)
    # print(path)
    updatePlot(path)

def updatePlot(path):
    global canvas
    for widget in canvas.get_tk_widget().winfo_children():
        widget.destroy()
    canvas.get_tk_widget().grid_forget()
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 35)
    ax.set_ylim(-10, 55)
    
    ax.plot(startx, starty, 'bo', markersize=10)
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((goalx1, goaly1), abs(goalx1 - goalx2), abs(goaly1 - goaly2), fill=None, edgecolor='black'))
    ax.plot(edgex, edgey)
    path = np.array(path)
    ax.scatter(path[:,0], path[:,1], color='r')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky='nsew')
startx, starty, startTheta, goalx1, goaly1, goalx2, goaly2, edgex, edgey = read.readEdge()
if goalx1 > goalx2:
    goalx1, goalx2 = goalx2, goalx1
if goaly1 > goaly2:
    goaly1, goaly2 = goaly2, goaly1
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
ax.set_xlim(-10, 35)
ax.set_ylim(-10, 55)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((goalx1, goaly1), abs(goalx1 - goalx2), abs(goaly1 - goaly2), fill=None, edgecolor='black'))
ax.plot(edgex, edgey)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky='nsew')

frame = tkinter.Frame(root)
frame.grid(row=2, column=0, columnspan=4)
trainButton = tkinter.Button(frame, text="Train", command=startTrain)
trainButton.grid(row=2, column=0, columnspan=2)

runButton = tkinter.Button(frame, text="Run", command=runfun)
runButton.grid(row=2, column=2, columnspan=2)

progress_bar = ttk.Progressbar(frame, orient='horizontal', length=200, mode='determinate')
progress_bar.grid(row=2, column=4, columnspan=4)

tkinter.mainloop()