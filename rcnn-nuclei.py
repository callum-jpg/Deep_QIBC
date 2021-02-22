import os
import sys
# import random
# import math
import numpy as np
import skimage.io # input/output
import skimage.color
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
MODEL_DIR = os.path.join(ROOT_DIR, 'samples', "logs")

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


#%% Work this out

for i in range(mask.shape[2]):
    temp = skimage.io.imread('test.jpg')
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(8,8))
    plt.imshow(temp)


#%% Testing with a DAPI image using coco class names. Obviously doesn't work since there's not an object class

# Convert dapi image to RGB
img = skimage.io.imread('dapi.TIF')

img = skimage.color.grey2rgb(img)

plt.imshow(img)

results = model.detect([img], verbose=1)

res = results[0]

visualize.display_instances(img, res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])



#%% Attempting to create new dataset class for reading masks required for training

class NucleiDataset(utils.Dataset):
    def load_nuclei(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("nuclei", 1, "nuclei")
        
        # Train or validation dataset?
        assert subset in ["train", "val"] # Test if condition is met
        dataset_dir = os.path.join(dataset_dir, subset)
        
#%%

# img_dir = os.listdir('test-mask')
# masks = os.listdir(os.listdir(os.path.join('test-mask', img_dir.pop(), 'masks')))

par_img = '1d4a5e729bb96b08370789cad0791f6e52ce0ffe1fcc97a04046420b43c851dd'

for i in os.listdir('test-mask'):
    masks = os.listdir(os.path.join('test-mask', i, 'masks'))
    
#%%
    
# Returns the image names (from parent folder name)
next(os.walk('test-mask'))[1]

class NucleiDataset(utils.Dataset):
    def load_nuclei(self, dataset_dir):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Define a new class, nucleus, which will be the name of identified masks
        self.add_class("nucleus", 1, "nucleus")
        
        # os.walk is a generator which returns a tuple of values: (current_path, 
        # directories in current_path, and files in current_path). os.walk recursively
        # looks through all directories that are available in the parent. Therefore, it
        # lists every file and folder.
        # os.walk()[1] returns the all the directories in the dataset_dir,
        # which in this case is a list of image names. 
        imgs = next(os.walk(dataset_dir))[1]
        
        for img in imgs:
            self.add_image(
                "nuclei",
                image_id=img, # Folder name (same as image name) becomes the unique ID
                # Path into the images and the source of the microscopy image
                path=os.path.join(dataset_dir, img, "images/{}.png".format(img))
                )
            
    def load_mask(self, image_id):
        
        info = self.image_info[image_id]
        
        # Specifically redirect to the masks subfolder
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        
        mask = []
        # Open all masks.png images in mask directory and store in array
        for file in next(os.walk(mask_dir))[2]:
            if file.endswith(".png"):
                # open and read the mask as a bool array
                individual_mask = skimage.io.imread(os.path.join(mask_dir, file)).astype(np.bool)
                # Add the opened mask to the mask list
                mask.append(individual_mask)
        
        # Stack joins arrays along a new axis. Different to concatenate which joins
        # them on the same axis
        mask = np.stack(mask, axis=-1)

        # Return the list of mask arrays and an array of class IDs. These are,
        # all 1's for however many masks were identified. [-1] selects the last
        # number in the shape 
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    # def image_reference(self, image_id):
    #     """Return the path of the image."""
    #     info = self.image_info[image_id]
    #     if info["source"] == "nucleus":
    #         return info["id"]
    #     else:
    #         super(self.__class__, self).image_reference(image_id)

#%%


data = NucleiDataset()

data.load_nuclei('test-mask')

data.prepare()

image_ids = data.image_ids

rag = range(10)

for i, j in enumerate(rag):
    #print(j)
    if i < int(len(rag) * 0.9):
        print(i, "training 90%")
        
for i, j in enumerate(rag):
    if i >= int(len(rag) * 0.9):
        print(i, "validation 10%")

#%% load images with masks on them to see if they're correct

for img in image_ids:
    image = data.load_image(img)
    mask, class_ids = data.load_mask(img)
    visualize.display_top_masks(image, mask, class_ids, data.class_names, limit=1)
    
    
#%%% dataset info

data = NucleiDataset()

data.load_nuclei('test-mask')

data.prepare()

print("Image Count: {}".format(len(data.image_ids)))
print("Class Count: {}".format(data.num_classes))
for i, info in enumerate(data.class_info):
    print("{:3}. {:50}".format(i, info['name']))

#%% Load a random image and its mask

import random

image_id = random.choice(data.image_ids)

img = data.load_image(image_id)

mask, class_ids = data.load_mask(image_id)

# Calculate the bounding box
bbox = utils.extract_bboxes(mask)

# Display image and instances
visualize.display_instances(img, bbox, mask, class_ids, data.class_names)



#%% Notes for using new NucleiDataset

import qibc_nucleus

config = qibc_nucleus.NucleusConfig()


img_dir = 'bw_train/'

data_train = qibc_nucleus.NucleiDataset()
data_train.load_nuclei(img_dir, "train")
data_train.prepare()

data_val = qibc_nucleus.NucleiDataset()
data_val.load_nuclei(img_dir, "val")
data_val.prepare()


# # Load and display random samples
# image_ids = np.random.choice(data_train.image_ids, 4)
# for image_id in image_ids:
#     image = data_train.load_image(image_id)
#     mask, class_ids = data_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, data_train.class_names, limit=1)
    

MODEL_DIR = 'logs'

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Load weights
model.load_weights('mask_rcnn_coco.h5', by_name=True, 
                    exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


DEVICE = "/cpu:0"

#%% Train

model.train(data_train, data_val,
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')


#%% Save model

model_path = os.path.join(MODEL_DIR, "mask_rcnn_nucleus2.h5")
model.keras_model.save_weights(model_path)


#%% testing model

import qibc_nucleus
import tensorflow as tf
import random

# WEIGHTS_PATH = 'logs/mask_rcnn_nucleus2.h5'
WEIGHTS_PATH = 'logs/pretrained.h5'

config = qibc_nucleus.NucleusConfig()

# __class__ is the instance to which this class instance belongs
# In this case, it's qibc_nucleus.NucleusConfig
class InferenceConfig(config.__class__):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    
DEVICE = "/cpu:0"

img_dir = 'bw_train/'
data_val = qibc_nucleus.NucleiDataset()
data_val.load_nuclei(img_dir, "val")
data_val.prepare()

# print("Images: {}\nClasses: {}".format(len(data_val.image_ids), data_val.class_names))

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
    
model.load_weights(WEIGHTS_PATH, by_name=True)

image_id = random.choice(data_val.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(data_val, config, image_id, use_mini_mask=False)
info = data_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       data_val.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])


# Run object detection
results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
    
# Run object detection
results = model.detect([image], verbose=1)

r = results[0]

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            data_val.class_names, r['scores'],
                            title="Predictions")
    

