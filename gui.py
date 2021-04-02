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
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import time

# import deepspace

import load_images
import detect


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
        # Create a list to hold image paths
        self.image_paths = []
        
        def browse_files():
            # Prompt user to select birectory
            self.foldername = filedialog.askdirectory()
            # Display selected folder
            self.lbl_explorer.configure(text="Folder Opened: "+self.foldername)
            # Clear the contents that may previously be displayed
            self.dir_contents.delete(0, tk.END)
            # For the user directory selected, search the directory
            # and print tif and png images into the ListBox
            for file in os.listdir(self.foldername):
                if file.endswith((".tif", ".png", ".TIF")):
                    self.dir_contents.insert(tk.END, file)

                    self.image_paths.append(os.path.join(self.foldername, file))

        
        def open_image_selection(_selection):
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
        self.dir_contents.bind("<Double-1>", open_image_selection)

        # Position the explorer and button within self.container
        self.lbl_explorer.grid(row = 0, column = 0)
        self.btn_browse.grid(row = 0, column = 1)
        self.dir_contents.grid(row = 1, column = 0)


class ChannelSelection:
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column, sticky='w')
    
        ### Building channel selection
        self.channel_options = [1, 2, 3, 4]
        
        # IntVar() to store number of channels selected
        self.channel_count = tk.IntVar()
        self.channel_count.set(self.channel_options[0])
        
        self.lbl_channel = tk.Label(self.container, 
                                          text = "Select number of channels:")
        
        

                # self.channel_box = tk.Label(self.container,
                #                             text = "channel {}".format(channel))
                # self.channel_box.grid(row = channel, column = 0)
        
        self.channel_option = tk.OptionMenu(self.container,
                                            self.channel_count,
                                            *self.channel_options,
                                            command = self.spawn_channels)
        
        # All of this channel stuff is messy. Perhaps all of these elements
        # should be stored in a dictionary? eg. dict["ch1"] would contain
        # all of the channel 1 entry fields in a list, with consistent indexing
        
        # Create channel entry fields that are by default disabled
        self.lbl_channel1 = tk.Label(self.container, text="Channel 1:") 
        # StringVar to hold entry field into
        self.channel1 = tk.StringVar() 
        # Trace to detect when entry field has changed
        self.channel1.trace("w", self.detect_check) 
        self.channel1_entry = tk.Entry(self.container,
                                       textvariable=self.channel1,
                                       state=tk.DISABLED)
        self.channel1_regex = tk.Entry(self.container, state=tk.DISABLED)
        
        self.lbl_channel2 = tk.Label(self.container, text="Channel 2:")
        self.channel2 = tk.StringVar() 
        self.channel2.trace("w", self.detect_check) 
        self.channel2_entry = tk.Entry(self.container, 
                                       textvariable = self.channel2,
                                       state=tk.DISABLED)
        self.channel2_regex = tk.Entry(self.container, state=tk.DISABLED)

        
        self.lbl_channel3 = tk.Label(self.container, text="Channel 3:")
        self.channel3 = tk.StringVar() 
        self.channel3.trace("w", self.detect_check) 
        self.channel3_entry = tk.Entry(self.container, 
                                 textvariable = self.channel3,
                                 state=tk.DISABLED)
        self.channel3_regex = tk.Entry(self.container, state=tk.DISABLED)
        
        self.lbl_channel4 = tk.Label(self.container, text="Channel 4:")
        self.channel4 = tk.StringVar() 
        self.channel4.trace("w", self.detect_check) 
        self.channel4_entry = tk.Entry(self.container, 
                                 textvariable = self.channel4,
                                 state=tk.DISABLED)
        self.channel4_regex = tk.Entry(self.container, state=tk.DISABLED)
        
        
        # Object channel selection
        self.object_channel = tk.IntVar(value=1)
        self.object_sel = tk.OptionMenu(self.container, self.object_channel, '')
        self.object_sel.config(state="disabled")
        self.object_sel.grid(row = 10, column=0)
#tk.OptionMenu(master, variable, value, values, kwargs)


        # Placing channel selection
        self.lbl_channel.grid(row = 2, column = 0, sticky="nsew")
        self.channel_option.grid(row = 2, column = 1, sticky="nsew")        

        self.lbl_channel1.grid(row=5, column=0, sticky="e")
        self.channel1_entry.grid(row=5, column=1, sticky="w")   
        self.channel1_regex.grid(row = 5, column = 2, stick = "w")
        
        self.lbl_channel2.grid(row=6, column=0, sticky="e")
        self.channel2_entry.grid(row=6, column=1, sticky="nsew")
        self.channel2_regex.grid(row = 6, column = 2, stick = "w")
        
        self.lbl_channel3.grid(row=7, column=0, sticky="e")
        self.channel3_entry.grid(row=7, column=1, sticky="nsew")
        self.channel3_regex.grid(row = 7, column = 2, stick = "w")
        
        self.lbl_channel4.grid(row=8, column=0, sticky="e")
        self.channel4_entry.grid(row=8, column=1, sticky="nsew")
        self.channel4_regex.grid(row = 8, column = 2, stick = "w")

    def detect_check(self, *args):
        """
        Trace does not accept app.detect as a command since its ran in __init__, 
        so this function is to pass trace information to the detection class. 
        """
        app.detect.detect_btn_state()

    def spawn_channels(self, _):
        """
        Running spawn_channels as a command from OptionMenu does actually
        return the selected value, but this function ignores the value as
        _ in order to use get() instead
        """
        # Get the selected channel by OptuionMenu (instead of using above _)
        channels = self.channel_count.get()
        object_dropdown = self.object_sel['menu']
        # Remove all old dropdown options
        self.object_sel['menu'].delete(0, tk.END)
        # Reset object to channel 1
        self.object_channel.set(1)
        
        if channels == 2:
            # Remove any text entered un unrequired channels
            self.channel3_entry.delete(0, tk.END)
            self.channel4_entry.delete(0, tk.END)
            self.channel3_regex.delete(0, tk.END)
            self.channel4_regex.delete(0, tk.END)
            # Enable or disable entry fields based on selection
            self.channel1_entry.config(state=tk.NORMAL)
            self.channel2_entry.config(state=tk.NORMAL)
            self.channel3_entry.config(state=tk.DISABLED)
            self.channel4_entry.config(state=tk.DISABLED)
            
            # Enable or disable relevent regex fields
            self.channel1_regex.config(state=tk.NORMAL)
            self.channel2_regex.config(state=tk.NORMAL)
            self.channel3_regex.config(state=tk.DISABLED)
            self.channel4_regex.config(state=tk.DISABLED)
            
            app.detect.btn_detection.config(state=tk.DISABLED)
            
            # Channels > 1 so decide object channel to run detection on
            self.object_sel.config(state="active")
            for ch in self.channel_options[0:2]:
                # Add a new value, ch, and also set the new dropdown value
                # of object_channel
                object_dropdown.add_command(label = ch, 
                                            command = lambda channel=ch: self.object_channel.set(channel))
            
        if channels == 3:
            self.channel4_entry.delete(0, tk.END)
            self.channel4_regex.delete(0, tk.END)
            
            self.channel1_entry.config(state=tk.NORMAL)
            self.channel2_entry.config(state=tk.NORMAL)
            self.channel3_entry.config(state=tk.NORMAL)
            self.channel4_entry.config(state=tk.DISABLED)

            self.channel1_regex.config(state=tk.NORMAL)
            self.channel2_regex.config(state=tk.NORMAL)
            self.channel3_regex.config(state=tk.NORMAL)
            self.channel4_regex.config(state=tk.DISABLED)

            self.object_sel.config(state="active")
            for ch in self.channel_options[0:3]:
                object_dropdown.add_command(label = ch, 
                                            command = lambda channel=ch: self.object_channel.set(channel))

        if channels == 4:
            self.channel1_entry.config(state=tk.NORMAL)
            self.channel2_entry.config(state=tk.NORMAL)
            self.channel3_entry.config(state=tk.NORMAL)
            self.channel4_entry.config(state=tk.NORMAL)
            
            self.channel1_regex.config(state=tk.NORMAL)
            self.channel2_regex.config(state=tk.NORMAL)
            self.channel3_regex.config(state=tk.NORMAL)
            self.channel4_regex.config(state=tk.NORMAL)
            
            self.object_sel.config(state="active")
            for ch in self.channel_options[0:4]:
                object_dropdown.add_command(label = ch, 
                                            command = lambda channel=ch: self.object_channel.set(channel))
            
        if channels == 1:
            # Delete info not requried
            self.channel1_entry.delete(0, tk.END)
            self.channel2_entry.delete(0, tk.END)
            self.channel3_entry.delete(0, tk.END)
            self.channel4_entry.delete(0, tk.END)
            
            self.channel1_regex.delete(0, tk.END)
            self.channel2_regex.delete(0, tk.END)
            self.channel3_regex.delete(0, tk.END)
            self.channel4_regex.delete(0, tk.END)
            
            self.channel1_entry.config(state=tk.DISABLED)
            self.channel2_entry.config(state=tk.DISABLED)
            self.channel3_entry.config(state=tk.DISABLED)
            self.channel4_entry.config(state=tk.DISABLED)
            
            self.channel1_regex.config(state=tk.DISABLED)
            self.channel2_regex.config(state=tk.DISABLED)
            self.channel3_regex.config(state=tk.DISABLED)
            self.channel4_regex.config(state=tk.DISABLED)
            
            self.object_sel.config(state="disabled")
                
class RunDetection:
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column, sticky='w')
        
        # Detection button
        self.btn_detection = tk.Button(self.container, 
                                       text = "Run Detection",
                                       state = "normal",
                                       command = self.run_detection)
        #self.btn_detection.config(state=tk.DISABLED)
        self.btn_detection.grid(row=0,column=0)
    
        
    # *args allows run_detection to accept the output from tk.trace
    def detect_btn_state(self, *args):
        """
        Require some logic for checking if detection button should be enabled
        
        run detection should be executed everytime the channel dropdown changes,
        everytime an entry field is filled/deleted
        
        In all cases, check that len(image_paths) > 1
        
        If channel_count == 1, check that len(image_paths) > 1. If True, 
        enable detection button
        
        If channel_count == 2, check that channel text is present and works on
        image_path extraction
        
        
        To check, channels should run the detection button check function
        """
        
        channels = app.channels.channel_count.get()
        # print(channels)
        
        # print(app.channels.channel1.get())

        if channels == 1:
            self.btn_detection.config(state=tk.NORMAL)

        elif (channels == 2 
            and len(app.channels.channel1.get()) > 0
            and len(app.channels.channel2.get()) > 0):
            self.btn_detection.config(state="normal")

            
            
        elif (channels == 3 
            and len(app.channels.channel1.get()) > 0
            and len(app.channels.channel2.get()) > 0
            and len(app.channels.channel3.get()) > 0):
            self.btn_detection.config(state="normal")

        elif (channels == 4 
            and len(app.channels.channel1.get()) > 0
            and len(app.channels.channel2.get()) > 0
            and len(app.channels.channel3.get()) > 0
            and len(app.channels.channel4.get()) > 0):
            self.btn_detection.config(state="normal")

        else: 
            self.btn_detection.config(state=tk.DISABLED)
        # if (len(app.channels.channel1.get()) == 0
        #     or len(app.channels.channel2.get()) == 0
        #     or len(app.channels.channel3.get()) == 0
        #     or len(app.channels.channel4.get()) == 0):
        #     self.btn_detection.config(state=tk.DISABLED)
        
        
        # if app.channels.channel_count.get() == 2:
        #     #self.btn_detection.config(state=tk.DISABLED)
        #     print("sdfsdf")
        
        # # Image paths
        # print(app.browse.image_paths)
        
        # # Number of channels 
        # print(app.channels.channel_count.get())
        
        
        # # # Channel info
        # print(app.channels.channel1.get())
        
        # print(app.channels.channel2.get())
        # print(len(app.channels.channel3.get()))
        
    def run_detection(self):
        self.queue = queue.Queue()
        
        # # Load images
        # images = load_images.LoadImages()
        # images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])
        # image_dir = "/home/think/Documents/deep-click/images"
        # images.load_images(image_dir)
        # # Run detection
        # nuclei_detection = detect.DetectNucleus()
        # # Select channel to run detection on (in this case, DAPI)
        # object_channel = images.channels[0]
        # nuclei_detection.run_detection(images.image_info, "high", "cpu", object_channel)
        # nuclei_detection.results
        
        # Gather unique channel filename parts
        unique_ch_names = list((app.channels.channel1.get(),
                               app.channels.channel2.get(),
                               app.channels.channel3.get(),
                               app.channels.channel4.get()))
        
        # Remove empty string from channels
        unique_ch_names = [ch for ch in unique_ch_names if ch]    
        
        images = load_images.LoadImages()
        images.add_channels(unique_ch_names)
        images.load_images(app.browse.foldername)
        
        if len(unique_ch_names) > 0:
            selected_obj = app.channels.object_channel.get()
            obj_channel = unique_ch_names[selected_obj - 1]
        else:
            obj_channel = None
            
        nuclei_detection = detect.DetectNucleus(self.queue,
                                                images.image_info, 
                                                "low", 
                                                "cpu", 
                                                obj_channel).start()


        #deepspace.DeepSpace(self.queue).start()
    
        
        self.parent.after(100, self.process_queue)

    def process_queue(self):
        try:
            msg = self.queue.get(0)
            print(msg)
            # Show result of the task if needed
            self.parent.after(100, self.process_queue)
        except queue.Empty:
            self.parent.after(100, self.process_queue)


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

    def increase(self):
        value = int(self.lbl_value["text"])
        self.lbl_value["text"] = value + 1
        app.print_iterator(int(self.lbl_value["text"]))

    def decrease(self): 
        value = int(self.lbl_value["text"])
        self.lbl_value["text"] = value - 1
        app.print_iterator(int(self.lbl_value["text"]))
        
        # Get strings of inputted channel name delieators
        channel_list = list((
            app.browse.channel1.get(),
            app.browse.channel2.get(),
            app.browse.channel3.get(),
            app.browse.channel4.get()))
        
        # Remove empty strings (ie. boxes not filled in)
        channel_list = [channel for channel in channel_list if channel]
        
        print(channel_list)

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

        self.lbl_display = tk.Label(self.container, text = str(data))
        self.lbl_display.grid(row=0, column=0)   
    
    def update(self, data):
        """
        Method to update the label created by this class
        """
        self.lbl_display["text"] = data
        
class ImageDisplay:
    def __init__(self, parent, row, column, image):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column)
        
        self.fig = Figure((5, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(image)
        
        # Change colour of plot background
        # It doesn't seem possible to make it transparent, so match it
        self.fig.patch.set_facecolor(color='lightgray')
        # Remove padding from around the plotted image
        self.fig.set_tight_layout(True)
        
        canvas = FigureCanvasTkAgg(self.fig, master=self.container)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)
        canvas._tkcanvas.grid(row=1, column=0)

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
        # Target changes the function thread.start() will run
        # Otherwise, it will look for a run() method
        Thread.__init__(self, target = self.hello)
        self.queue = queue

    def hello(self):
        for i in range(5):
            time.sleep(0.5)
            self.queue.put(i)





# class IntergalacticWidget:
#     """
#     A test widget for running a function from another file in a thread
#     """
#     def __init__(self, parent, row, column):
#         self.parent = parent
#         self.container = tk.Frame(self.parent)
#         self.container.grid(row = row, column = column)
        
#         btn_start = tk.Button(self.container, text = "Start", command = self.start)
#         btn_start.grid(row=1, column=1, sticky="nsew", padx = 20, pady = 20) 
        
#     def start(self):
#         self.queue = queue.Queue()

#         deepspace.DeepSpace(self.queue).start()
        
#         self.parent.after(100, self.process_queue)

#     def process_queue(self):
#         try:
#             msg = self.queue.get(0)
#             print(msg)
#             # Show result of the task if needed
#             self.parent.after(100, self.process_queue)
#             app.print_iterator1(msg)
#         except queue.Empty:
#             self.parent.after(100, self.process_queue)

class DeepQIBCgui(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        self.master = master
        
        self.browse = BrowseFiles(self.master, 0, 0)
        
        browse1 = BrowseFiles(self.master, 0, 1)
        
        self.channels = ChannelSelection(self.master, 1, 0)

        iterator1 = Iterator(self.master, 1, 1)
        
        self.detect = RunDetection(self.master, 2, 0)

        # Test image
        x = np.zeros((200, 200, 3))
        ImageDisplay(self.master, 0, 2, x)

        # intergalactic = IntergalacticWidget(self.master, 5, 1)
        
        
        # Create a display obect at 1x2 displaying value 0
        self.iterator = IteratorDisplay(self.master, 1, 2, 0)
        
    def print_iterator(self, data):
        "Function trggered by a button press"
        self.iterator.update(data)

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









