import os
import sys
# import random
# import math
import numpy as np
import skimage.io # input/output
# import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")

# import warnings
# warnings.filterwarnings("ignore")

# Importing mask RCNN
sys.path.append(ROOT_DIR) # Append local version of rcnn
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# For testing, use the COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

# store logs
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Trained weights file (testing with COCO for now)
MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

# Image directory
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

#%% Inference class

class InferenceConfig(coco.CocoConfig):
    # Batch size = 1. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()
config.display()

#%% building model

model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

model.load_weights("mask_rcnn_coco.h5", by_name=True)

# List of coco class names to categorise masks
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#%% Testing model

img = skimage.io.imread('test.jpg')

plt.imshow(img)

results = model.detect([img], verbose=1)

res = results[0]

visualize.display_instances(img, res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])

#%% 

# This is access to the identified masks
# axis2 = the number of masks (eg objects). 480=height, 640=width
res['masks'].astype(int).shape

# Store the masks
mask = res['masks'].astype(int)
# numpyify it


#%% Work this out 25-22-20

for i in range(mask.shape[2]):
    temp = skimage.io.imread('test.jpg')
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(8,8))
    plt.imshow(temp)










