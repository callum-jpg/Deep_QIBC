import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set WD to Mask_RCNN/samples/nucleus
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import nucleus
import cv2

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")


#%%

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0" 

#%% 
import nucleus


# Inference Configuration
config = nucleus.NucleusInferenceConfig()

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
    
# Load weights
weights_path = os.path.join(ROOT_DIR, 'samples/nucleus/mask_rcnn_nucleus2.h5')
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

import time
start_time = time.time()


data_dir = os.path.join(ROOT_DIR, "datasets/nucleus/lil-test")

data = nucleus.NucleusDataset1()
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

# In img_array, 0th element, 1st element is the image array
visualize.display_instances(img_array[0][1], res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])

plt.imshow(cv2.cvtColor(img_array[0][1], cv2.COLOR_RGB2GRAY), cmap="gray")
plt.axis("off")

#%% Display nuclei image + masked image

import cv2

# Create subplots with two axes (ncol=2)
fig, ax = plt.subplots(ncols=2)

# add the grayscale image to ax[0]
# cv2.cvtColor to convert RGB image fed into mask RCNN back into grayscale
ax[0].imshow(cv2.cvtColor(img_array[0][1], cv2.COLOR_RGB2GRAY), cmap="gray")
ax[0].axis("off")

# Empty class names so they don't appear on image.
# display_instances has a mandatory class_ids argument
class_names = ['', '']

# Add the RCNN ROI, masks etc. image to ax[1]
visualize.display_instances(img_array[0][1], 
                            res['rois'], res['masks'], res['class_ids'], class_names,
                            show_bbox=False,
                            ax=ax[1])

 
#%%

# For the above results.append, extract as such
# [0].. for the first image analysed
# ..[1].. for the model detect data ([0] if you wanted filename)
# ..[0] To enter the list created by Mask RCNN
# ['rois'] to select a required dict key, in this case 'rois'
results[0][1][0]['rois']

# Alternatively, find the masks
results[0][1][0]['masks'].shape

###### [-1] did that do something??


# Filename extracted as:
results[0][0]


#%% Can you label a single Mask-RCNN mask with ndimage.label?

three = np.arange(60).reshape((3, 4, 5))

i, j, k = np.indices(three.shape)


#%% This works for extracting and labelling a single mask

from scipy import ndimage

# The masks generated is a 3d array, with k = each individual mask
nuclei_masks = results[0][1][0]['masks']

# Elipses extendeds the selection to the third dimension of the array,
# without it, you'd just slice i, j and no k. Mask information 
# from RCNN is stored in k
# Select the first identified mask, k=0, and preserve i, j (image dimensions)
mask_to_lab = nuclei_masks[...,0]

# Label the single mask as before
labelled_mask, num = ndimage.label(mask_to_lab, np.ones((3,3)))

# 
fig, ax = plt.subplots(ncols=3, nrows=1, dpi=300)
# Plot input image
ax[0].imshow(cv2.cvtColor(img_array[0][1], cv2.COLOR_RGB2GRAY), cmap="gray")
ax[0].axis("off")
# Plot single ndiamge.labelled mask from mask r-cnn
ax[1].imshow(labelled_mask, cmap='gray')
ax[1].axis("off")

# label all of the masks from r-cnn 
visualize.display_instances(img_array[0][1], 
                            res['rois'], res['masks'], res['class_ids'], class_names,
                            show_bbox=False,
                            ax=ax[2])


#%% Transpose 3d array into 1d

from skimage.color import label2rgb

# Extract the mask information from a single image
nuclei_masks = results[0][1][0]['masks']

# Create an empty array of the image shape
combined_arr = np.zeros(nuclei_masks.shape[0:2], dtype=bool)

# Iterate through all k arrays and add these arrays to combined_arr
# This adds TRUE in the place where a nuclei is found by mask rcnn
for i in range(0, nuclei_masks.shape[2]):
    combined_arr = combined_arr + nuclei_masks[...,i]
    
# Here, ndimage.label labels the identified nuclei (eg those which are TRUE). 
# I don't like this though - it is removing control of the RCNN definition of
# a nuclei which could be accidentally overwritten by ndimage.label.
# It is probably best to create a labelling function which labels 
# nuclei BEFORE they are consolodated into combined_arr. Will have to work out
# how to preserve the counting up of nuclei. Then, can be labelled and plotted
# as before. 
labelled_mask, num = ndimage.label(combined_arr, np.ones((3,3)))

fig, ax = plt.subplots(ncols=3, nrows=1, dpi=300)
# Plot input image
ax[0].imshow(cv2.cvtColor(img_array[0][1], cv2.COLOR_RGB2GRAY), cmap="gray")
ax[0].axis("off")
# Plot single ndiamge.labelled mask from mask r-cnn
ax[1].imshow(label2rgb(labelled_mask, bg_label=0), cmap="gray", vmin=0, vmax=125)
ax[1].axis("off")

# label all of the masks from r-cnn 
visualize.display_instances(img_array[0][1], 
                            res['rois'], res['masks'], res['class_ids'], class_names,
                            show_bbox=False,
                            ax=ax[2])



#%% plot results

res = results[0]

class_names = ['BG','nucleus']

visualize.display_instances(img, res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])

#%%
### TODO ###
# add model.train into a function
# Extract mask arrays in a list

# Mask R-CNN identifies nuclei in individual arrays - How to integrate this?

# Attempt to see how ndimage.label labels identified nuclei. Maybe it can
# incrementally label nuclei masks in a list of arrays 

# Find a way to display coloured nuclei masks onto image with matplotlib and numpy
# not mask r-cnn visualise

# Can you combine multiple mask arrays for one image by multiplying them?
# Yes you can. ndimage.label might lead overright some genuine nuclei in identitiy though
a = np.array([1,1,0])
b = np.array([0,1,1])
a*b
