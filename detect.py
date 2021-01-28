import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage

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
        
class AdjustNucleusConfigMed(nucleus.NucleusInferenceConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MIN_DIM = 1024
        IMAGE_MAX_DIM = 1024
        
        
class AdjustNucleusConfigHigh(nucleus.NucleusInferenceConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = "pad64"


class DetectNucleus:
    def __init__(self):
        # Instantiate the desired computation config
        self.empty = []
        self.config_low = AdjustNucleusConfigLow()
        self.config_med = AdjustNucleusConfigMed()
        self.config_high = AdjustNucleusConfigHigh()
        
    def run_detection(self, computation_requirement=None, device=None):
        
        # Load optional configs
        if computation_requirement == "low":
            config = self.config_low
        if computation_requirement == "med":
            config = self.config_med
        if computation_requirement == "high":
            config = self.config_high
        else:
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
        
        # Load images
        # NEED TO ADD OPTIONS FOR GIVING DIRECTORY/CHANNELS
        images = load_images.LoadImages()
        
        images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])
        
        image_dir = "images"
        
        images.load_images(image_dir)
        
        results = [] 
        for img in images.image_info:
            print(img['w1DAPI']) # returns the image name
            # img_rgb = skimage.color.gray2rgb(img['w1DAPI'][1])
            # results.append([img['w1DAPI'][0], model.detect([img['w1DAPI'][1]], verbose=1)])
            results.append([img['w1DAPI'][0], model.detect([img['w1DAPI'][1]], verbose=1)])
        
        return results
        
        
  
            
            
            

        
#%%

DetectNucleus().run_detection()


#%% testing

DEVICE = "/cpu:0" 

config = DetectNucleus().config_low

# Test class varaibles with DetectNucleus().config_low.IMAGE_RESIZE_MODE
# or config.IMAGE_RESIZE_MODE

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)


#%% Will it blend?
import skimage
from mrcnn import visualize


DEVICE = "/cpu:0" 


# Inference Configuration
config = AdjustNucleusConfigLow()

#config = nucleus.NucleusInferenceConfig()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
    
# Load weights
print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)

# results = [] 
# for img in images.image_info:
#     print(img['w1DAPI']) # returns the image name
#     # img_rgb = skimage.color.gray2rgb(img['w1DAPI'][1])
#     results.append([img['w1DAPI'][0], model.detect([
#         skimage.color.gray2rgb(img['w1DAPI'][1])], verbose=1)])

images = load_images.LoadImages()

images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

image_dir = "images"

images.load_images(image_dir)


results = [] 
for img in images.image_info:
    print(img['w1DAPI']) # returns the image name
    # img_rgb = skimage.color.gray2rgb(img['w1DAPI'][1])
    # results.append([img['w1DAPI'][0], model.detect([img['w1DAPI'][1]], verbose=1)])
    results.append([img['w1DAPI'][0], model.detect([img['w1DAPI'][1]], verbose=1)])
    
res1 = results[0][1][0]
    
# In img_array, 0th element, 1st element is the image array
#display_image = skimage.color.gray2rgb(images.image_info[0]['w1DAPI'][1])

class_names = ['','']

visualize.display_instances(images.image_info[0]['w1DAPI'][1], 
                            res1['rois'], res1['masks'], res1['class_ids'], class_names,
                            show_bbox=False)


    
# plt.imshow(skimage.color.rgb2gray(images.image_info[0]['w1DAPI'][1]), cmap="gray")
gray_image = skimage.color.rgb2gray(images.image_info[0]['w1DAPI'][1])
plt.imshow(gray_image, cmap="gray")
plt.axis("off")




#%%
for i in images.image_info:
    print(i['w1DAPI'][0], i['w1DAPI'][1])


#%%

import nucleus1 # Contains NucleusDataset1

DEVICE = "/cpu:0" 

# Inference Configuration
# config = nucleus.NucleusInferenceConfig()

config = AdjustNucleusConfigLow()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
    
# Load weights
print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)

import time
start_time = time.time()

data_dir = os.path.join("datasets/nucleus/lil-test")

data = nucleus1.NucleusDataset1()
img_array = data.load_nucleus(data_dir)
# img_arr is a list of tuples. [0] is the image name, [1] is the cv2 image array

# Below iterates through the image array and adds the image name and 
# model.detect output to a list
results = [] 
for img in img_array:
    print(img) # returns the image name
    results.append([img[0], model.detect([img[1]], verbose=1)])
    
print("--- {} seconds ---".format(time.time() - start_time))


res = results[0][1][0]

class_names = ['BG','nucleus']

# # In img_array, 0th element, 1st element is the image array
visualize.display_instances(img_array[0][1], res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])

plt.imshow(cv2.cvtColor(img_array[0][1], cv2.COLOR_RGB2GRAY), cmap="gray")
plt.axis("off")


#%% Investigating image shape
import skimage

img_array[0][1].shape

skimage.color.gray2rgb(images.image_info[0]['w1DAPI'][1]).shape
        
        