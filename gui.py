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
import pandas as pd


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import time

import load_images
import detect
import visualise
import process_images

import deepspace


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
                                # Enable detection button 
            # Check button state
            app.detect.detect_btn_state()

        
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
        
class SaveData:
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column)
        
        # Empty string that will be updated with the selected save directory, if desired
        self.foldername = []
        
        # Textbox to display selected directory
        self.lbl_explorer = tk.Label(
            master = self.container, 
            text = "Select a data save directory", 
            width = 75,
            bg = "white")
        
        # Browse files button
        #browse = tk.Frame(master = window)
        self.btn_browse = tk.Button(
            master = self.container, 
            width = 20, 
            text = "Browse",
            command = self.browse_folders)

        # Position the explorer and button within self.container
        self.lbl_explorer.grid(row = 0, column = 0)
        self.btn_browse.grid(row = 0, column = 1)
        
    def browse_folders(self):
        """
        Browse folders for a save directory
        """
        self.foldername = filedialog.askdirectory()
        if len(self.foldername):
            # Display selected folder to save data
            self.lbl_explorer.configure(text="Saving data in: "+self.foldername)


        

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
        self.lbl_obj_channel = tk.Label(self.container, text="Select object channel:")
        self.object_sel = tk.OptionMenu(self.container, self.object_channel, '')
        self.object_sel.config(state="disabled")
        self.object_sel.grid(row = 10, column=1, sticky="w")
        self.lbl_obj_channel.grid(row=10, column=0, sticky="e")
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
        
        # Store detection results
        self.detection_results = []
        
        # Detection button
        self.btn_detection = tk.Button(self.container, 
                                       text = "Run Detection",
                                       state = tk.DISABLED,
                                       command = self.run_detection)
        
        self.lbl_detection = tk.Label(self.container, text="Select img directory")

        # Computation level selection
        self.lbl_computation = tk.Label(self.container, text = "Select computation level")
        self.computation_level = tk.StringVar(value = "low")
        self.computation_selection = tk.OptionMenu(self.container,
                                                   self.computation_level,
                                                   *["low", "med", "high"])  
        
        # GPU/CPU selection
        self.lbl_cpu_gpu = tk.Label(self.container, text = "Select device to run deepQIBC on")
        self.cpu_gpu = tk.StringVar(value = "cpu")
        self.cpu_gpu_selection = tk.OptionMenu(self.container,
                                               self.cpu_gpu,
                                               *["cpu", "gpu"]) 
          
                
        self.computation_selection.grid(row = 0, column = 0, sticky="nsew")
        self.lbl_computation.grid(row = 0, column = 1, sticky="w")
        
        self.cpu_gpu_selection.grid(row = 1, column = 0, sticky="nsew")
        self.lbl_cpu_gpu.grid(row = 1, column = 1, sticky="w")
        
        self.btn_detection.grid(row=2, column=0, sticky="e")
        self.lbl_detection.grid(row=2, column=1, sticky="w")
    
        
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

        if (channels == 1
            and app.browse.dir_contents.size()):
            self.btn_detection.config(state=tk.NORMAL)
            self.lbl_detection.configure(text="")

        elif (channels == 2 
            and len(app.channels.channel1.get()) > 0
            and len(app.channels.channel2.get()) > 0):
            self.btn_detection.config(state="normal")
            self.lbl_detection.configure(text="")

        elif (channels == 3 
            and len(app.channels.channel1.get()) > 0
            and len(app.channels.channel2.get()) > 0
            and len(app.channels.channel3.get()) > 0):
            self.btn_detection.config(state="normal")
            self.lbl_detection.configure(text="")

        elif (channels == 4 
            and len(app.channels.channel1.get()) > 0
            and len(app.channels.channel2.get()) > 0
            and len(app.channels.channel3.get()) > 0
            and len(app.channels.channel4.get()) > 0):
            self.btn_detection.config(state="normal")
            self.lbl_detection.configure(text="")

        elif not app.browse.dir_contents.size(): 
            print("hello there")
            self.btn_detection.config(state=tk.DISABLED)
            self.lbl_detection.configure(text="No imgs found in selected folder")
        
    def run_detection(self):
        self.queue = queue.Queue()
        
        # Gather unique channel filename parts
        unique_ch_names = list((app.channels.channel1.get(),
                               app.channels.channel2.get(),
                               app.channels.channel3.get(),
                               app.channels.channel4.get()))
    

        # Remove empty string from channels
        unique_ch_names = [ch for ch in unique_ch_names if ch]    
        
        if unique_ch_names:
            channels = unique_ch_names
        else:
            channels = None
        
        images = load_images.LoadImages()
        images.add_channels(unique_ch_names)
        images.load_images(app.browse.foldername)
        
        if len(unique_ch_names) > 0:
            selected_obj = app.channels.object_channel.get()
            # Get delineating string that corresponds to the object channel
            obj_channel = unique_ch_names[selected_obj - 1]
        else:
            obj_channel = None
            
        nuclei_detection = detect.DetectNucleus()
        
        nuclei_detection.run_detection(images.image_info, 
                                        #"low", 
                                        self.computation_level.get(),
                                        #"cpu", 
                                        self.cpu_gpu.get(),
                                        obj_channel)
        
        self.detection_results.append(nuclei_detection.results)
        
        # Load and label masks
        labelled = process_images.ProcessMasks()
        
        labelled.label_masks(nuclei_detection.results)
        
        # print(labelled.labelled_masks[0]['masks'].max())
        
        intensity = process_images.RecordIntensity(images.image_info, 
                                                   labelled.labelled_masks,
                                                   channels)
        
        # Add itentified masks to image_info
        intensity.consolidate_masks_with_images(obj_channel)

        data = intensity.record_intensity()
        
        # Pass data to data visualisation widget
        app.data_vis.update_data(data)
        
        # Save data if directory selected
        if len(app.save_data.foldername):
            df = pd.DataFrame(data=data)
            df.to_csv("deepQIBC_data.csv", index=False)
            
        
        
        ### Fixing threading?
        ## https://stackoverflow.com/questions/25351488/intermittent-python-thread-error-main-thread-is-not-in-main-loop
        
    #     # Instantiate detection class        
    #     self.detection_thread = detect.DetectNucleus_thread(self.queue)
        
    #     # Create a Thread object with the desired target function to run
    #     # args are passed to the function
    #     self.detection_thread = Thread(target = self.detection_thread.run_detection,
    #                                         args = (images.image_info, 
    #                                                 "low", 
    #                                                 "cpu", 
    #                                                 obj_channel,))
    #     # Start the thread
    #     self.detection_thread.start()
    
    #     # Begin polling to check data from thread
    #     self.detecting = self.parent.after(1000, self.process_queue)
        
    #     # Counter for tracking images processed and when to clear after command
    #     self.detection_counter = len(images.image_info)

    # def process_queue(self):
    #     """
    #     process_queue is used to recieve the data from the threaded detection.
    #     self.queue.get(0) returns the latest result and self.parent.after
    #     allows for checking of new data in the queue.
        
    #     When all data has been processed, the after command is cancelled with
    #     after_cancel
    #     """
    #     try:
    #         msg = self.queue.get(0)
    #         #print(msg)
    #         # Show result of the task if needed
    #         self.detecting = self.parent.after(1000, self.process_queue)
    #         self.detection_counter -= 1
    #     except queue.Empty:
    #         self.detecting = self.parent.after(1000, self.process_queue)
    #     if self.detection_counter == 0:
    #         print("Queue cleared!")
    #         self.parent.after_cancel(self.detecting)
    #         #self.detection_thread.exit()


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
        # value = int(self.lbl_value["text"])
        # self.lbl_value["text"] = value + 1
        # app.print_iterator(int(self.lbl_value["text"]))
        
        rollover_value = 4
        value = int(self.lbl_value["text"])
        self.lbl_value["text"] = (value + 1) % rollover_value
        app.print_iterator(int(self.lbl_value["text"]))

    def decrease(self): 
        # value = int(self.lbl_value["text"])
        # self.lbl_value["text"] = value - 1
        # app.print_iterator(int(self.lbl_value["text"]))
        
        rollover_value = 4
        value = int(self.lbl_value["text"])
        self.lbl_value["text"] = (value - 1) % rollover_value
        app.print_iterator(int(self.lbl_value["text"]))
        
        print(app.detect.detection_results)
        

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
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column, sticky = "n")
        
        # Create buttons to iterate through plotted images
        btn_next = tk.Button(self.container, text=">>", command=self.next_img)
        btn_next.grid(row=2, column=1, sticky="nsew")        
        btn_prev = tk.Button(self.container, text="<<", command=self.prev_img)
        btn_prev.grid(row=2, column=0, sticky="nsew")    
        
        # Create a placeholder image before detection is run
        placeholder_img = np.zeros((512, 512, 3))
        
        self.fig = plt.Figure((4, 4))
        self.ax = self.fig.add_subplot(111)
        # Initisialise with an empty, placeholder image
        self.ax.imshow(placeholder_img)
        
        # Why not add object and detection figures into subplots of the same figure?
        # I preferred how creating two separate figures looked and it seems to work well.
        # Though a future change could be to see if a nicer solution can be created
        # using subplots
        
        # Remove padding from around the plotted image
        self.fig.set_tight_layout(True)
        
        # Change colour of plot background
        # It doesn't seem possible to make it transparent, so match it
        self.fig.patch.set_facecolor(color='lightgray')

        # Object channel canvas
        self.canvas_object = FigureCanvasTkAgg(self.fig, master=self.container)
        self.canvas_object.draw()
        self.canvas_object.get_tk_widget().grid(row=0, column=0)
        self.canvas_object._tkcanvas.grid(row=1, column=0)
        
        # Detection output canvas
        self.canvas_detection = FigureCanvasTkAgg(self.fig, master=self.container)
        self.canvas_detection.draw()
        self.canvas_detection.get_tk_widget().grid(row=0, column=0)
        self.canvas_detection._tkcanvas.grid(row=1, column=1)
        
        # Store the index of image to show
        self.slideshow_index = 0
        
    def update_data(self, data):
        """Add the detection data"""
        self.detection_data = data
    
    # def update_plot(self, index):
    #     """Change the index of plot to show"""
    #     self.plot(index)
            
        
    def plot(self, index):
        """Plot the desired result following running of the RunDetection class
        
        index: an int that is found in the range of RunDetection.results"""
        # Accessing the results - slices in order
        # [0][0] enter the first list of image data (masks, rois, etc.)
        # [1] enters the image specific data ([0] would be image filename[0] + image array[1])
        # [0] iamge data is a dict in a list, so this enters that first dict
        # 'masks' or whatever else to read the desired data   
        source_img = self.detection_data[0][index][0][1]
        masks_img = self.detection_data[0][index][1][0]['masks']  

        image = visualise.colour_masks(source_img, masks_img)

        # Object channel image
        self.fig_object = plt.Figure((4, 4))
        self.ax_object = self.fig_object.add_subplot(111)
        self.ax_object.imshow(source_img)
        self.ax_object.title.set_text("Source image")
                
        # Detection image
        self.fig_detection = plt.Figure((4, 4))
        self.ax_detection = self.fig_detection.add_subplot(111)
        self.ax_detection.imshow(image)
        self.ax_detection.title.set_text("Detected objects")
        
        # Remove padding from around the plotted image
        self.fig_object.set_tight_layout(True)
        self.fig_detection.set_tight_layout(True)
        
        # Change colour of plot background
        # It doesn't seem possible to make it transparent, so match it
        self.fig_object.patch.set_facecolor(color='lightgray')
        self.fig_detection.patch.set_facecolor(color='lightgray')

        # Clear previous canvas
        #self.canvas.get_tk_widget().pack_forget()
        self.canvas_clear(self.canvas_object)
        self.canvas_clear(self.canvas_detection)
        
        # Object channel canvas
        self.canvas_object = FigureCanvasTkAgg(self.fig_object, master=self.container)
        self.canvas_object.draw()
        self.canvas_object.get_tk_widget().grid(row=0, column=0)
        self.canvas_object._tkcanvas.grid(row=1, column=0)
        
        # Detection output canvas
        self.canvas_detection = FigureCanvasTkAgg(self.fig_detection, master=self.container)
        self.canvas_detection.draw()
        self.canvas_detection.get_tk_widget().grid(row=0, column=0)
        self.canvas_detection._tkcanvas.grid(row=1, column=1)
        
    def canvas_clear(self, canvas):
        """"
        Delete previously plotted canvas
        
        canvas: a FigureCanvasTkAgg object
        """
        for item in canvas.get_tk_widget().find_all():
            print("deleting:", item)
            canvas.get_tk_widget().delete(item)

    def next_img(self):        
        """Next img"""
        # Check the length of image plotting data
        num_imgs = len(self.detection_data[0])
        # Rolling over index to first image
        current_index = self.slideshow_index
        self.slideshow_index = (current_index + 1) % num_imgs
        self.plot(self.slideshow_index)
        
    def prev_img(self):
        """Previous img"""
        # Check the length of image plotting data
        num_imgs = len(self.detection_data[0])
        # Rolling over index to first image
        current_index = self.slideshow_index
        self.slideshow_index = (current_index - 1) % num_imgs
        self.plot(self.slideshow_index)
        


class DataVisualisation:
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column)
        
        # Label above axes option menus
        self.lbl_axis = tk.Label(self.container, text = "Select plotting axes:")
        
        
        # Labels for axes dropdowns
        self.lbl_x = tk.Label(self.container, text = "X-axis:")
        self.lbl_y = tk.Label(self.container, text = "Y-axis:")
        self.lbl_colour = tk.Label(self.container, text = "Point colour:")
        
        # StringVar to hold column titles to be plotted
        self.x_value = tk.StringVar(value = "NA")
        self.y_value = tk.StringVar(value = "NA")
        self.colour_value = tk.StringVar(value = "NA")
    
        
        self.x_option = tk.OptionMenu(self.container,
                                      self.x_value, 
                                      "",
                                      command = self.update_plot)
        
        self.y_option = tk.OptionMenu(self.container,
                                      self.y_value, 
                                      "",
                                      command = self.update_plot)
                
        self.colour_option = tk.OptionMenu(self.container,
                                      self.colour_value, 
                                      "",
                                      command = self.update_plot)
        
        self.lbl_axis.grid(row = 1, column = 1, sticky="nsew")
        
        self.lbl_x.grid(row = 2, column = 0, sticky="e")     
        self.x_option.grid(row = 2, column = 1, sticky="nsew")  
        
        self.lbl_y.grid(row = 3, column = 0, sticky="e") 
        self.y_option.grid(row = 3, column = 1, sticky="nsew")   

        self.lbl_colour.grid(row = 4, column = 0, sticky="e") 
        self.colour_option.grid(row = 4, column = 1, sticky="nsew")  
        
        
        ## Editing plot limits
        # Min/max label
        self.lbl_min = tk.Label(self.container, text = "min")
        self.lbl_max = tk.Label(self.container, text = "max")
        self.lbl_min.grid(row = 1, column = 2, sticky="nsew")
        self.lbl_max.grid(row = 1, column = 3, sticky="nsew")
        
        # register allows us to execute the validate_input function upon user input
        # self.validate is then used as a callback function
        self.validate = self.container.register(self.validate_input)
        
        # x limits
        # min/max must be string var to allow for floats
        self.x_min = tk.StringVar(value = None)
        self.entry_x_min = tk.Entry(self.container, textvariable= self.x_min,
                                    validate = "key", 
                                    validatecommand=(self.validate, "%P"))
        
        self.entry_x_min.grid(row = 2, column = 2, ipady=2)
      
        self.x_max = tk.StringVar(value = None)
        self.entry_x_max = tk.Entry(self.container, textvariable= self.x_max)
        self.entry_x_max.grid(row = 2, column = 3, ipady=2)
        
        # y limits
        self.y_min = tk.StringVar(value = None)
        self.entry_y_min = tk.Entry(self.container, textvariable= self.y_min)
        self.entry_y_min.grid(row = 3, column = 2, ipady=2)
      
        self.y_max = tk.StringVar(value = None)
        self.entry_y_max = tk.Entry(self.container, textvariable= self.y_max)
        self.entry_y_max.grid(row = 3, column = 3, ipady=2)
        
        # colour limits
        self.colour_min = tk.StringVar(value = None)
        self.entry_colour_min = tk.Entry(self.container, textvariable= self.colour_min)
        self.entry_colour_min.grid(row = 4, column = 2, ipady=2)
      
        self.colour_max = tk.StringVar(value = None)
        self.entry_colour_max = tk.Entry(self.container, textvariable= self.colour_max)
        self.entry_colour_max.grid(row = 4, column = 3, ipady=2)
        
        # Update plot limits button
        self.btn_update_limits = tk.Button(self.container, text = "Update plot limits",
                                           command = self.update_plot_limits)
        self.btn_update_limits.grid(row = 5, column = 2, columnspan = 2)
        
        # Simple placeholder data
        x = np.arange(0, 4*np.pi, 0.1)
        y = np.sin(x)
        
        self.fig = plt.Figure((4, 4))
        self.ax = self.fig.add_subplot(111)
        # Initisialise with an empty, placeholder image
        #self.ax.imshow(placeholder_img)
        self.ax.plot(x, y)
        
        # Remove padding from around the plotted image
        self.fig.set_tight_layout(True)
        
        # Change colour of plot background
        # It doesn't seem possible to make it transparent, so match it
        self.fig.patch.set_facecolor(color='lightgray')

        # Object channel canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.container)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.canvas._tkcanvas.grid(row=0, column=1, columnspan = 3)
            
    def validate_input(self, x):
        """Validate that user input is a float"""
        try:
            # Check if x is either length 0 or a float
            x == "" or float(x)
            return True
        except:
            return False

    def update_data(self, data):
        """Receives data from the RunDetection class and updates the 
        dropdown accordingly
        
        data: RunDetection data following intensity.consolidate_masks_with_images
        and intensity.record_intensity transformations"""
        
        self.detection_data = data
        
        # Slice from [2:] since columns 0:2 are image number and object number
        self.detection_data_cols = [k for k in self.detection_data.keys()][2:]
        
        # Set some default values for x and y to plot
        self.x_value.set(self.detection_data_cols[0])
        self.y_value.set(self.detection_data_cols[1])
        self.colour_value.set("None")
        
        # Plot 2nd (x) and 3rd (y, colour) coloumns
        self.plot(self.detection_data[self.x_value.get()], 
                  self.detection_data[self.y_value.get()])
        
        # Set min/max axis values based on what is selected by matplotlib
        self.x_min.set(self.ax.get_xlim()[0])
        self.x_max.set(self.ax.get_xlim()[1])
        self.y_min.set(self.ax.get_ylim()[0])
        self.y_max.set(self.ax.get_ylim()[1])
        
        # self.colour_min.set(self.ax.get_clim()[0])
        # self.colour_max.set(self.ax.get_clim()[1])
        #print(self.detection_data[self.x_value.get()])
        
        # Clear all old dropdown options from previous detections
        self.x_option['menu'].delete(0, tk.END)
        self.y_option['menu'].delete(0, tk.END)
        self.colour_option['menu'].delete(0, tk.END)
        
        # In essence, the tk._setit function allows for a new value to be added
        # to the OptionMenu 'menu' variable and to call the command
        # self.update_plot when the new value (col) as an argument. _setit
        # accepts three arguments: the variable, the value to be added (col) and
        # the command to be executed upon value selection. It's lovely
        
        # _setit is required as add_command used here overwrites previous
        # commands issued to the OptionMenu. 
        
        # Every time a new option is selected in the OptionMenu, update_plot
        # is called to update the data visualisation. 
        for col in self.detection_data_cols:
                self.x_option['menu'].add_command(label = col,
                                                  command = tk._setit(self.x_value, col, self.update_plot))
                
                self.y_option['menu'].add_command(label = col,
                                                  command = tk._setit(self.y_value, col, self.update_plot))
                
        # Add "None" to the list of options for point colour
        for col in ["None"] + self.detection_data_cols:
                self.colour_option['menu'].add_command(label = col,
                                                  command = tk._setit(self.colour_value, col, self.update_plot)) 

    def update_plot(self, _):
        """Reads currently selected x, y and colour and updates the plot"""
        print("updating...")
        self.plot(self.detection_data[self.x_value.get()],
                  self.detection_data[self.y_value.get()],
                  (self.detection_data[self.colour_value.get()]) if self.colour_value.get() != "None" else None)

        # Set min/max axis values based on what is selected by matplotlib
        self.x_min.set(self.ax.get_xlim()[0])
        self.x_max.set(self.ax.get_xlim()[1])
        self.y_min.set(self.ax.get_ylim()[0])
        self.y_max.set(self.ax.get_ylim()[1])

        if self.colour_value.get() != "None":
            self.colour_min.set(min(self.detection_data[self.colour_value.get()]))
            self.colour_max.set(max(self.detection_data[self.colour_value.get()]))
        # If None is reselected, clear the entry to nothing
        elif self.colour_value.get() == "None":
            self.colour_min.set(value = "")
            self.colour_max.set(value = "")
            
        
    def update_plot_limits(self):
        """Following update plot axes button press, update the plot with new limits"""
        
        print(self.x_min.get())
        x_mm = [float(self.x_min.get()), self.x_max.get()]
        y_mm = [self.y_min.get(), self.y_max.get()]
        colour_mm = [self.colour_min.get(), self.colour_max.get()]
        
        self.plot(self.detection_data[self.x_value.get()],
                  self.detection_data[self.y_value.get()],
                  (self.detection_data[self.colour_value.get()]) if self.colour_value.get() != "None" else None,
                  x_mm, 
                  y_mm,
                  colour_mm)
                
    
    def plot(self, x, y, colour = None, xmm=None, ymm=None, cmm=None):
        """plot QIBC data"""
        print("plotting")
        
        print(xmm, ymm, cmm)
        
        # Object channel image
        self.fig = plt.Figure((4, 4))
        self.ax = self.fig.add_subplot(111)
        
        self.data_plot = self.ax.scatter(x = x, 
                        y = y, 
                        c = colour)
        # Add x and y axis based on user selection
        self.ax.set_xlabel(self.x_value.get())
        self.ax.set_ylabel(self.y_value.get())
        
        # Colour legend
        if colour != None: self.fig.colorbar(self.data_plot, shrink=.3, pad=0.05, aspect=10)

        # Remove padding from around the plotted image
        self.fig.set_tight_layout(True)
        
        # Change colour of plot background
        # It doesn't seem possible to make it transparent, so match it
        self.fig.patch.set_facecolor(color='lightgray')

        # Clear previous canvas
        #self.canvas.get_tk_widget().pack_forget()
        self.canvas_clear(self.canvas)
        
        # Object channel canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.container)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.canvas._tkcanvas.grid(row=0, column=1, columnspan = 3)
        
        
    def canvas_clear(self, canvas):
        """"Delete previously plotted canvas"""
        for item in canvas.get_tk_widget().find_all():
            print("deleting:", item)
            canvas.get_tk_widget().delete(item)

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


class IntergalacticWidget:
    """
    A test widget for running a function from another file in a thread
    """
    def __init__(self, parent, row, column):
        self.parent = parent
        self.container = tk.Frame(self.parent)
        self.container.grid(row = row, column = column)
        
        btn_start = tk.Button(self.container, text = "Start", command = self.start)
        btn_start.grid(row=1, column=1, sticky="nsew", padx = 20, pady = 20) 
        
        btn_start = tk.Button(self.container, text = "STOP ME", command = self.cancel)
        btn_start.grid(row=1, column=2, sticky="nsew", padx = 20, pady = 20) 
        
        self.shuttle_vector = 5

        
    def start(self):
        self.queue = queue.Queue()
        
        self.counter = self.shuttle_vector
    
        deepspace.DeepSpace(self.queue, self.shuttle_vector).start()
        
        self.test = self.parent.after(100, self.process_queue_intergalactic)

    def process_queue_intergalactic(self):
        try:
            msg = self.queue.get(0)
            print(msg)
            # Show result of the task if needed
            #self.parent.after(100, self.process_queue_intergalactic)
            self.test = self.parent.after(100, self.process_queue_intergalactic)
            app.print_iterator1(msg)
            self.counter -= 1
        except queue.Empty:
            #self.parent.after(100, self.process_queue_intergalactic)
            self.test = self.parent.after(100, self.process_queue_intergalactic)
        if self.counter == 0:
            self.parent.after_cancel(self.test)
    
        # self.parent.after_cancel(self.test)
        
    def cancel(self):
        self.parent.after_cancel(self.test)
            

class DeepQIBCgui(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        self.master = master
        
        self.show_image = False
        
        self.browse = BrowseFiles(self.master, 1, 0)
        
        self.channels = ChannelSelection(self.master, 1, 1)
        
        self.save_data = SaveData(self.master, 2, 0)
        
        self.detect = RunDetection(self.master, 2, 1)     
            
        self.display = ImageDisplay(self.master, 0, 0)
        
        self.data_vis = DataVisualisation(self.master, 0, 1)

        # self.intergalactic = IntergalacticWidget(self.master, 5, 1)
        
        # Create a display obect at 1x2 displaying value 0
        # self.iterator = IteratorDisplay(self.master, 2, 2, 0)

        self.master.after(1000, self.display_image)
        
    def display_image(self):
        """
        Display initial detection data
        
        This solution works OK for the first run, but needs to be changed.
        The show_image bool could be replaced by a more intelligent 'checker'
        function that checks if the detection data has changed or not. Currently,
        there is no way to change show_image back to false, thus data can only
        be updated once.
        """
        if len(self.detect.detection_results) > 0 and self.show_image == False:
            print("plot!")
            self.display.update_data(self.detect.detection_results)
            # Show the 0th index plot
            self.display.plot(0)
            self.show_image = True

        print("checking results...")
        self.master.after(1000, self.display_image)

    def print_iterator(self, data):
        "Function trggered by a button press"
        print("this is data:", data)
        self.iterator.update(data)

    # def print_iterator1(self, data):
    #     disp = IteratorDisplay(self.master, 2, 1, data)
    
        
if __name__ == "__main__":
    root = tk.Tk()
    app = DeepQIBCgui(root)
    root.mainloop()

