import tkinter as tk

window = tk.Tk()

for i in range(3):
    for j in range(3):
        frame = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
        frame.grid(row=i, column=j, padx=5, pady=5) # External padding
        label = tk.Label(master=frame, text=f"Row {i}\nColumn {j}")
        label.pack(padx=5, pady=5) # Internal padding

window.mainloop()


#%%
window = tk.Tk()

window.rowconfigure(0, minsize=50)
window.columnconfigure([0, 1, 2, 3], minsize=50)

label1 = tk.Label(text="1", bg="black", fg="white")
label2 = tk.Label(text="2", bg="black", fg="white")
label3 = tk.Label(text="3", bg="black", fg="white")
label4 = tk.Label(text="4", bg="black", fg="white")

label1.grid(row=0, column=0)
label2.grid(row=0, column=1, sticky="ew")
label3.grid(row=0, column=2, sticky="ns")
label4.grid(row=0, column=3, sticky="nsew") # nsew fills up the space (50x50)
# sticky=nsew is like pack() fill=tk.BOTH 

window.mainloop()

#%% Handling events

window = tk.Tk()

def handle_click(event):
    print("The button was clicked!")

button = tk.Button(text="Click me!")
button.pack()

button.bind("<Button-1>", handle_click)

window.mainloop()

#%%

window = tk.Tk()

def increase():
    value = int(lbl_value["text"])
    lbl_value["text"] = value + 1


def decrease():
    value = int(lbl_value["text"])
    lbl_value["text"] = value - 1

window.rowconfigure(0, minsize=50, weight=1)
window.columnconfigure([0, 1, 2], minsize=50, weight=1)

btn_decrease = tk.Button(master=window, text="-", command=decrease)
btn_decrease.grid(row=0, column=0, sticky="nsew")

lbl_value = tk.Label(master=window, text="0")
lbl_value.grid(row=0, column=1)

btn_increase = tk.Button(master=window, text="+", command=increase)
btn_increase.grid(row=0, column=2, sticky="nsew")

window.mainloop()


#%% Building DeepQIBC GUI

import tkinter as tk
from tkinter import filedialog

window = tk.Tk()
window.title("DeepQIBC")

def browsefiles():
    filename = filedialog.askopenfilename()
    lbl_explorer.configure(text="File Opened: "+filename)

# Textbox to display selected directory
lbl_explorer = tk.Label(master = window, text = "File explorer", width = 20)

# Browse files button
#browse = tk.Frame(master = window)
btn_browse = tk.Button(master = window, width = 20, text = "Browse",
                       command = browsefiles)


# Layout
lbl_explorer.grid(row = 0, column = 0)
btn_browse.grid(row = 0, column = 1)



#%% Class GUI
# https://dev.to/abdurrahmaanj/building-an-oop-calculator-and-what-it-means-to-write-a-widget-library-4560

import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image
from PIL import ImageTk
from threading import Thread
import queue

import time

class BrowseFiles:
    """
    BrowseFiles widget
    """
    def __init__(self, parent, row, column):
        self.parent = parent
        # Create a frame called container
        # This container will hold the BrowseFiles widget
        self.container = tk.Frame(self.parent)
        # Place this container at the given row/column
        # This will be the parent window passed to an element/widget (eg. a button)
        # contained within. 
        self.container.grid(row = row, column = column)
        
        def browse_files():
            # Prompt user to select birectory
            self.filename = filedialog.askdirectory()
            # Display selected folder
            self.lbl_explorer.configure(text="Folder Opened: "+self.filename)
            # Clear the contents that may previously be displayed
            self.dir_contents.delete(0, tk.END)
            # For the user directory selected, search the directory
            # and print tif and png images into the ListBox
            for file in os.listdir(self.filename):
                if file.endswith((".tif", ".png")):
                    self.dir_contents.insert(tk.END, file)
        
        def get_image_selection(_selection):
            img_selection = self.dir_contents.curselection()
            self.img_filename = self.dir_contents.get(img_selection)
            img_open = Image.open(self.img_filename)
            width, height = img_open.size
            aspect = width / height
            # Round leads to approximation of the height, but it should
            # not distort the image too much, hopefully...
            img_resized = img_open.resize((500, round(500/aspect)))      
            img = ImageTk.PhotoImage(img_resized)
            new_window = tk.Toplevel()
            new_window.title(self.img_filename)   
            lab = tk.Label(new_window, image = img)
            lab.grid(row = 0, column = 0)
            new_window.mainloop()
            
        # Textbox to display selected directory
        self.lbl_explorer = tk.Label(
            master = self.container, 
            text = "File explorer", 
            width = 75,
            bg = "white")
        
        # Browse files button
        #browse = tk.Frame(master = window)
        self.btn_browse = tk.Button(
            master = self.container, 
            width = 20, 
            text = "Browse",
            command = browse_files)
        
        # List box to display the image files found in the selected dir
        self.dir_contents = tk.Listbox(
            self.container,
            width = 75)
        # When a file in dir_contents is double clicked, open the image
        self.dir_contents.bind("<Double-1>", get_image_selection)

        # Position the explorer and button within self.container
        self.lbl_explorer.grid(row = 0, column = 0)
        self.btn_browse.grid(row = 0, column = 1)
        self.dir_contents.grid(row = 1, column = 0)
        
class Iterator:
    """
    For testing passing data between widgets/classes
    """
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column)
        
        btn_decrease = tk.Button(self.container, text="-", command=self.decrease)
        btn_decrease.grid(row=0, column=0, sticky="nsew")
        
        self.lbl_value = tk.Label(self.container, text="0")
        self.lbl_value.grid(row=0, column=1)
        
        btn_increase = tk.Button(self.container, text="+", command=self.increase)
        btn_increase.grid(row=0, column=2, sticky="nsew")    

        btn_trigger = tk.Button(self.container, text = "!!!", command = self.trigger)
        btn_trigger.grid(row=1, column=1, sticky="nsew") 
        
        self.list = []

    def increase(self):
        value = int(self.lbl_value["text"])
        self.lbl_value["text"] = value + 1
        app.print_iterator(int(self.lbl_value["text"]))
        # Trying to work out how to detect a list has changed between classes

    def decrease(self): 
        value = int(lbl_value["text"])
        self.lbl_value["text"] = value - 1
        app.print_iterator(int(self.lbl_value["text"]))
        
    def trigger(self):
        #l = ListIterator()
        self.queue = queue.Queue()
        ListIterator(self.queue).start()
        
        self.parent.after(100, self.process_queue)


        
    def process_queue(self):
        try:
            msg = self.queue.get(0)
            print(msg)
            # Show result of the task if needed
            self.parent.after(100, self.process_queue)
            app.print_iterator1(msg)
        except queue.Empty:
            self.parent.after(100, self.process_queue)

class IteratorDisplay:
    def __init__(self, parent, row, column, data):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column)
        
        lbl_display = tk.Label(self.container, text = str(data))
        lbl_display.grid(row=0, column=0)   
     

class ListIterator(Thread):
    """
    Class to mimic the detection of deepQIBC. 
    
    I'm trying to work out if I can display the detections as they 
    are measured.
    
    Therefore, I want to pass self.list 5 times to the master window, 
    rather than just passing the completed list.
    
    Should I run it in a different thread? I think so
    """
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    # class method called "run" to enable threading
    def run(self):
        for i in range(5):
            time.sleep(0.5)
            self.queue.put(i)




class DeepQIBCgui(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        self.master = master
        
        browse = BrowseFiles(self.master, 0, 0)
        
        browse1 = BrowseFiles(self.master, 0, 1)
        
        iterator = Iterator(self.master, 1, 0)

        iterator = Iterator(self.master, 1, 1)
        
        
    def print_iterator(self, data):
        disp = IteratorDisplay(self.master, 1, 2, data)

    def print_iterator1(self, data):
        disp = IteratorDisplay(self.master, 2, 1, data)

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepQIBCgui(root)
    root.mainloop()


#%% Testing others code

import queue
from tkinter import ttk

class GUI:
    def __init__(self, master):
        self.master = master
        self.test_button = tk.Button(self.master, command=self.tb_click)
        self.test_button.configure(
            text="Start", background="Grey",
            padx=50
            )
        self.test_button.pack(side=tk.TOP)
        
        self.test_button1 = tk.Button(self.master)
        self.test_button1.configure(
            text="it does nothing", background="Grey",
            padx=50
            )
        self.test_button1.pack(side=tk.BOTTOM)


    def progress(self):
        self.prog_bar = ttk.Progressbar(
            self.master, orient="horizontal",
            length=200, mode="indeterminate"
            )
        self.prog_bar.pack(side=tk.TOP)

    def tb_click(self):
        self.progress()
        self.prog_bar.start()
        self.queue = queue.Queue()
        ThreadedTask(self.queue).start()
        self.master.after(100, self.process_queue)

    def process_queue(self):
        try:
            # This is entered when the queue is not empty. 
            # Here, message is the "Task finished" message
            msg = self.queue.get(0)
            print("queue get", msg)
            self.master.after(100, self.process_queue)
            #self.prog_bar.stop()
        except queue.Empty:
            # If the queue is empty, test it again after 100 seconds
            self.master.after(100, self.process_queue)

class ThreadedTask(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
    def run(self):
        #time.sleep(2)
        for i in range(5):
            time.sleep(1)
            self.queue.put(i)
        # Put an item into the quene
        # This is executed after the sleep function has run
        #self.queue.put("Task finished") 

root = tk.Tk()
root.title("Test Button")
main_ui = GUI(root)
root.mainloop()









