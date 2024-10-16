import tkinter
import os
import train  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

filename = ""

def readFile(file):
    # Reading the file and extracting data
    with open('NN_HW1_DataSet/basic/' + file) as f:
        X = []
        y = []
        for line in f:
            l = line.split()
            x = []
            
            for i in range(len(l)):
                if i < len(l) - 1:
                    x.append(float(l[i]))
                else:
                    y.append(int(l[i]))
            X.append(x)
                

    lr = lrText.get("1.0", tkinter.END).strip()  # Remove trailing newlines
    if lr == "":
        lr = 0.1
    epoch = epochText.get("1.0", tkinter.END).strip()  # Remove trailing newlines
    if epoch == "":
        epoch = 100
    accuracy = accuracyText.get("1.0", tkinter.END).strip()  # Remove trailing newlines
    if accuracy == "":
        accuracy = 100
    # Create the plot using the train module
    fig = train.create_plot(X, y, float(lr), int(epoch), int(accuracy))
    update_plot(fig)

# Function to update the plot on the canvas
def update_plot(fig):
    # Clear the canvas before adding the new figure
    global canvas
    for widget in canvas.get_tk_widget().winfo_children():
        widget.destroy()  # Clear the canvas content
    canvas.get_tk_widget().grid_forget()  # Remove the old canvas widget
    canvas = FigureCanvasTkAgg(fig, master=root)  # Recreate the canvas with the new figure
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky='nsew')  # Grid layout

def updateFileName(f):
    global file
    file.set(f)
    global filename
    filename = f
    print(filename)

# Setup the Tkinter window
dir = os.listdir("NN_HW1_DataSet/basic")
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# Make the grid layout more flexible by configuring row/column weights
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create the file menu
file_menu = tkinter.Menu(root)
menu = tkinter.Menu(file_menu)
for i in dir:
    menu.add_command(label=i, command=lambda f=i: updateFileName(f))
file_menu.add_cascade(label="File", menu=menu)
root.config(menu=file_menu)

file = tkinter.StringVar()
file.set("No file selected")
fileLabel = tkinter.Label(root, textvariable=file, font=("Times",15,"bold"))
fileLabel.grid(row=0, column=0, columnspan=4, pady=10)  # Grid layout

# Create an initial empty figure and canvas
fig, ax = plt.subplots()  # Initial empty plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky='nsew')  # Grid layout

# Learning rate and epoch input fields
lrlabel = tkinter.Label(root, text="Enter the learning rate:", font=("Times",15,"bold"))
lrlabel.grid(row=2, column=0, padx=5, pady=5)  # Grid layout

lrText = tkinter.Text(root, width=5, height=1, font=("Times",15,"bold"))
lrText.insert("1.0","0.1")
lrText.grid(row=2, column=1, padx=5, pady=5)  # Grid layout

epochlabel = tkinter.Label(root, text="Enter the number of epochs:", font=("Times",15,"bold"))
epochlabel.grid(row=2, column=2, padx=5, pady=5)  # Grid layout

epochText = tkinter.Text(root, width=5, height=1, font=("Times",15,"bold"))
epochText.insert("1.0", "100")
epochText.grid(row=2, column=3, padx=5, pady=5)  # Grid layout

accuracyLabel = tkinter.Label(root, text="Stop when accuracy reach(%)", font=("Times",15,"bold"))
accuracyLabel.grid(row=3, column=0, columnspan=1, pady=10)  # Grid layout

accuracyText = tkinter.Text(root, width=30, height=1, font=("Times",15,"bold"))
accuracyText.insert("1.0", 100)
accuracyText.grid(row=3, column=1, columnspan=3, pady=10)  # Grid layout

# Start button to generate plot
btn = tkinter.Button(root, text="start", command=lambda: readFile(filename), font=("Times",15,"bold"))
btn.grid(row=4, column=0, columnspan=4)  # Grid layout

# Start the Tkinter main loop
tkinter.mainloop()
