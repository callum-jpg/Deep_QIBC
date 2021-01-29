import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage
import time

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
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = 1024
        IMAGE_MAX_DIM = 1024       
        
class AdjustNucleusConfigHigh(nucleus.NucleusConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Min max not available with pad64
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MIN_SCALE = 2


class DetectNucleus:
    def __init__(self):
        # Instantiate the desired computation config
        self.config_low = AdjustNucleusConfigLow()
        self.config_med = AdjustNucleusConfigMed()
        self.config_high = AdjustNucleusConfigHigh()
        self.results = []
        
    def run_detection(self, images, object_channel, 
                      computation_requirement=None, device=None):
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
        else:
            print("Using low settings")
            config = self.config_low

        # Load preferred device
        if device == "cpu":
            DEVICE = "/cpu:0"
        if device == "gpu":
            DEVICE = "/GPU:0"
        else:
            DEVICE = "/cpu:0"
            
            
        with tf.device(DEVICE): 
                model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
                
        
        print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
        model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)
        
        start_time = time.time()
        
        for img in images:
            print(img[object_channel]) # returns the image name
            self.results.append([img[object_channel][0], 
                                 model.detect([img[object_channel][1]], verbose=1)])
            
        print("--- {} seconds ---".format(time.time() - start_time))

    

# #%%

# # Load images
# images = load_images.LoadImages()

# images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

# image_dir = "images"

# images.load_images(image_dir)

# # Run detection
# nuclei_detection = DetectNucleus()

# # Select channel to run detection on (in this case, DAPI)
# object_channel = images.channels[0]

# nuclei_detection.run_detection(images.image_info, object_channel, "low", "cpu")

# nuclei_detection.results

# #%%

# from mrcnn import visualize

# # images = load_images.LoadImages()

# # images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

# # image_dir = "images"

# # images.load_images(image_dir)

# res1 = nuclei_detection.results[0][1][0]
    
# # In img_array, 0th element, 1st element is the image array
# #display_image = skimage.color.gray2rgb(images.image_info[0]['w1DAPI'][1])

# class_names = ['','']

# visualize.display_instances(images.image_info[0]['w1DAPI'][1], 
#                             res1['rois'], res1['masks'], res1['class_ids'], class_names,
#                             show_bbox=False)


    
# # plt.imshow(skimage.color.rgb2gray(images.image_info[0]['w1DAPI'][1]), cmap="gray")
# gray_image = skimage.color.rgb2gray(images.image_info[0]['w1DAPI'][1])
# plt.imshow(gray_image, cmap="gray")
# plt.axis("off")
