import tkinter
import os
import perceptron  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# Global lists to store data
x = []
y = []
c = []

# Function to read the file and update x, y, c
def readFile(file):
    x.clear()
    y.clear()
    c.clear()
    # Reading the file and extracting data
    with open('NN_HW1_DataSet/basic/' + file) as f:
        for line in f:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
            c.append(int(line.split()[2]))
    
    # Create the plot using the perceptron module
    fig = perceptron.create_plot(x, y, c, file)
    update_plot(fig)

# Function to update the plot on the canvas
def update_plot(fig):
    # Clear the canvas before adding the new figure
    global canvas
    canvas.get_tk_widget().pack_forget()  # Remove the old canvas widget
    canvas = FigureCanvasTkAgg(fig, master=root)  # Recreate the canvas with the new figure
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

# Setup the Tkinter window
dir = os.listdir("NN_HW1_DataSet/basic")
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# Create the file menu
file_menu = tkinter.Menu(root)
menu = tkinter.Menu(file_menu)
for i in dir:
    menu.add_command(label=i, command=lambda f=i: readFile(f))
file_menu.add_cascade(label="File", menu=menu)
root.config(menu=file_menu)

# Create an initial empty figure and canvas

fig, ax = plt.subplots()  # Initial empty plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)


# Start the Tkinter main loop
tkinter.mainloop()
