import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage
import time
# from threading import Thread

mrcnn_dir = os.path.abspath("Mask_RCNN")
sys.path.append(mrcnn_dir)
import mrcnn.model as modellib

# Nucleus directory
nuc_dir = os.path.abspath("Mask_RCNN/samples/nucleus")
sys.path.append(nuc_dir)
import nucleus

import load_images

# Path to nucleus trained weights
NUCLEUS_TRAINED_WEIGHTS = 'weights/mask_rcnn_nucleus.h5'

# Logs dir
LOGS_DIR = "logs"

# Mask R-CNN allows for the alteration of class variables (eg GPU_COUNT) to
# change settings for the pipeline. Below are three classes (low, med, high)
# that change class variables to alter the computation requirement as desired. 
class AdjustNucleusConfigLow(nucleus.NucleusConfig):
        # Set batch size to 1 to run one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Smaller number = less overlaps
        RPN_NMS_THRESHOLD = 0.7
        # "square" resizes image into a square image with dimensions
        # [IMAGE_MIN_DIM, IMAGE_MAX_DIM], with the shorter side padded up with zeros
        # This allows for the computational load to be reduced by making the images smaller
        # "pad64" will pad the input image to multiples of 64 with zeros
        # "pad64" does not resize the image whereas square does
        # IMAGE_MIN/MAX_DIM is the resized image dimension
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        
class AdjustNucleusConfigMed(nucleus.NucleusConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Smaller number = less overlaps
        RPN_NMS_THRESHOLD = 0.7
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = 1024
        IMAGE_MAX_DIM = 1024   

        
class AdjustNucleusConfigHigh(nucleus.NucleusConfig):
    # High config doesn't seem to add much, if anything, above med. Delete?
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Smaller number = less overlaps
        RPN_NMS_THRESHOLD = 0.5
        
        
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = 2048
        IMAGE_MAX_DIM = 2048  

# My attempt at threading. Only works for one run then breaks
# class DetectNucleus(Thread):
#     def __init__(self, queue, images, computation_requirement="low", device="cpu",
#                  object_channel=None):
#         # Target changes the function thread.start() will run
#         # Otherwise, it will look for a run() method
#         Thread.__init__(self, target = self.run_detection)
#         self.queue = queue
#         self.images = images
#         self.computation_requirement = computation_requirement
#         self.device = device
#         self.object_channel = object_channel
        
#         # Instantiate the desired computation config
#         self.config_low = AdjustNucleusConfigLow()
#         self.config_med = AdjustNucleusConfigMed()
#         self.config_high = AdjustNucleusConfigHigh()
#         self.results = []
        
#     # def run_detection(self, images, computation_requirement="low", device="cpu",
#     #                   object_channel=None,):
#     def run_detection(self):
#         """
#         Runs the Mask R-CNN model detection. Images is from images.image_info.
        
#         object_channel: the image that the masks will be generated 
#         for (eg. DAPI-stained nuclei).
        
#         computation_requirement: low/med/high will determine the intensity 
#         that the model is run at. Low settings are less accurate but less
#         computationally intensive than high settings.
        
#         device: select cpu or gpu for running the model
#         """
        
#         # Load optional configs
#         if self.computation_requirement == "low":
#             print("Using low settings")
#             config = self.config_low
#         if self.computation_requirement == "med":
#             print("Using medium settings")
#             config = self.config_med
#         if self.computation_requirement == "high":
#             print("Using high settings")
#             config = self.config_high

#         # Load preferred device
#         if self.device == "cpu":
#             DEVICE = "/cpu:0"
#         if self.device == "gpu":
#             DEVICE = "/GPU:0"
#         else:
#             DEVICE = "/cpu:0"
            
#         if self.object_channel is None:
#             print("No object channel given. Using default.")
#             object_channel = "image_data"
            
            
#         with tf.device(DEVICE): 
#                 model = modellib.MaskRCNN(mode="inference",
#                               model_dir=LOGS_DIR,
#                               config=config)
                
        
#         print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
#         model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)
        
#         start_time = time.time()
        
#         for img in self.images:
#             print(img[object_channel]) # returns the image name
#             self.results.append([img[object_channel][0], 
#                                  model.detect([img[object_channel][1]], verbose=1)])
#             self.queue.put([img[object_channel][0], 
#                                  model.detect([img[object_channel][1]], verbose=1)])
            
#         print("--- {} seconds ---".format(time.time() - start_time))

 


# class DetectNucleus_thread:
#     def __init__(self, queue):
#         # Instantiate the desired computation config
#         self.config_low = AdjustNucleusConfigLow()
#         self.config_med = AdjustNucleusConfigMed()
#         self.config_high = AdjustNucleusConfigHigh()
#         self.results = []
        
#     def run_detection(self, images, computation_requirement="low", device="cpu",
#                       object_channel=None,):
#         """
#         Runs the Mask R-CNN model detection. Images is from images.image_info.
        
#         object_channel: the image that the masks will be generated 
#         for (eg. DAPI-stained nuclei).
        
#         computation_requirement: low/med/high will determine the intensity 
#         that the model is run at. Low settings are less accurate but less
#         computationally intensive than high settings.
        
#         device: select cpu or gpu for running the model
#         """
        
#         # Load optional configs
#         if computation_requirement == "low":
#             print("Using low settings")
#             config = self.config_low
#         if computation_requirement == "med" or "Med" or "Medium":
#             print("Using medium settings")
#             config = self.config_med
#         if computation_requirement == "high":
#             print("Using high settings")
#             config = self.config_high

#         # Load preferred device
#         if device == "cpu":
#             DEVICE = "/cpu:0"
#         if device == "gpu":
#             DEVICE = "/GPU:0"
#         else:
#             DEVICE = "/cpu:0"
            
#         if object_channel is None:
#             print("No object channel given. Using default.")
#             object_channel = "image_data"
            
            
#         with tf.device(DEVICE): 
#                 model = modellib.MaskRCNN(mode="inference",
#                               model_dir=LOGS_DIR,
#                               config=config)
                
        
#         print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
#         model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)
        
#         start_time = time.time()
        
#         for img in images:
#             print(img[object_channel]) # returns the image name
#             self.results.append([img[object_channel][0], 
#                                  model.detect([img[object_channel][1]], verbose=0)])
#             # Put the detection data into the queue to be received by the tkinter GUI
#             self.queue.put([img[object_channel][0], 
#                                  model.detect([img[object_channel][1]], verbose=1)])
            
#         print("--- {} seconds to run detection ---".format(time.time() - start_time))
        

class DetectNucleus:
    def __init__(self):
        # Instantiate the desired computation config
        self.config_low = AdjustNucleusConfigLow()
        self.config_med = AdjustNucleusConfigMed()
        self.config_high = AdjustNucleusConfigHigh()
        self.results = []
        
    def run_detection(self, images, computation_requirement="low", device="cpu",
                      object_channel=None,):
        """
        Runs the Mask R-CNN model detection. Images is from images.image_info.
        
        object_channel: the image that the masks will be generated 
        for (eg. DAPI-stained nuclei).
        
        computation_requirement: low/med/high will determine the intensity 
        that the model is run at. Low settings are less accurate but less
        computationally intensive than high settings.
        
        device: select cpu or gpu for running the model
        """
        
        # Load optional configs
        if computation_requirement == "low":
            print("Using low settings")
            config = self.config_low
        if computation_requirement == "med":
            print("Using medium settings")
            config = self.config_med
        if computation_requirement == "high":
            print("Using high settings")
            config = self.config_high

        # Load preferred device
        if device == "cpu":
            DEVICE = "/cpu:0"
        if device == "gpu":
            DEVICE = "/GPU:0"
        else:
            DEVICE = "/cpu:0"
            
        if object_channel is None:
            print("No object channel given. Using default.")
            object_channel = "image_data"
            
            
        with tf.device(DEVICE): 
                model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
                
        
        print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
        model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)
        
        start_time = time.time()
        
        for img in images:
            print(img[object_channel][0]) # returns the image name
            self.results.append([img[object_channel], 
                                 model.detect([img[object_channel][1]], verbose=1)])
            
        print("--- {} seconds to run detection ---".format(time.time() - start_time))


