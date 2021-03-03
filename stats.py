# Gathering useful stats to test model

#%% Calculating IoU

import tensorflow as tf

import qibc_nucleus
import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

mrcnn_dir = os.path.abspath("Mask_RCNN")
sys.path.append(mrcnn_dir)
import mrcnn.model as modellib
import mrcnn.utils as utils
import mrcnn.visualize as visualize
from mrcnn.config import Config

import skimage
import cv2

#%%

class AdjustNucleusConfigHigh(Config):
    """Configuration for training on the nucleus segmentation dataset.
    Edited from Waleed's nucleus implementation of mask RCNN"""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


config = AdjustNucleusConfigHigh()
#config = qibc_nucleus.NucleusInferenceConfig()

#%
# img_dir = 'datasets/nucleus/bw_train/'
img_dir = 'datasets/nucleus/label_test/'

dataset = qibc_nucleus.NucleiDataset()
dataset.load_nuclei(img_dir) 
dataset.prepare()

image_id = random.choice(dataset.image_ids)

# You give the dataset object to load_image_gt. It is the load_image_gt
# function which then runs dataset.load_mask to load all available masks
# as defined by your load_masks function in qibc_nucleus.
# These gt masks are later matched with the inference masks and scores
# calculated
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    
# Convert to 8-bit
# Since model was trained on 8-bit DSB18 images but metamorph are 16-bit,
# convert to 8-bit to improve score
out = np.zeros(image.shape, dtype=np.uint8)

out = cv2.normalize(image, out, 0, 255, cv2.NORM_MINMAX)

image = out
    
info = dataset.image_info[image_id]

# This doesn't seem to work for high resolution images
# This is because load_image_gt resizes images and requries the input 
# dimensions to be multiples of 64. The resize dimensions is declared by 
# the config
# Thus, utils.resize_image is scaling the gt image down to 512x512
# and then running detection on this smaller image. 
# Moreover, the molded images are normalised (in addition to resized) for 
# input into the neural network
# GT masks load fine, but prediction isn't accurate. 
# Moreover, the image loads weird too - perhaps due to RGB conversions

# The above statements are false. Mistakes in resizing were due to "crop"
# IMAGE_RESIZE_MODE being selected. 
# The GT image 'looking weird' was due to shortened histogram. Consider
# playing around with skimage.io stuff
# 
# However, this method leads to no object detection for some reason - why?
# Testing with the DSB18 images works fine, though these were used to train
# so probably not a good idea to use at all for this.
# UPDATE: it seems to be something to do with image resolution. Perhaps 
# due to the training taking place on 512x512 images, when a larger image, 
# eg metamorph 1040x1392




#% Exploring dataset

mask, class_ids = dataset.load_mask(image_id)

dataset.load_mask

test = dataset.load_image(image_id)

#%  Load model

LOGS_DIR = os.path.join("logs")

weights = 'weights/mask_rcnn_nucleus.h5'

DEVICE = '/cpu:0'

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)


model.load_weights(weights, by_name=True)

#%%

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax



#%%

print("imageID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

# Run object detection
results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

# Display results
r = results[0]
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# Compute AP over range 0.5 to 0.95 and print it
# AP stands for average precision, as detailed here: https://cocodataset.org/#detection-eval
utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                       r['rois'], r['class_ids'], r['scores'], r['masks'],
                       verbose=1)

mAP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                 r['rois'], r['class_ids'], r['scores'], r['masks'])


visualize.display_differences(
    image,
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    dataset.class_names, ax=get_ax(),
    show_box=False, show_mask=False,
    iou_threshold=0.5, score_threshold=0.5)




#%%

# compute_matches links predicted masks with GT masks
# gt_match is an array of indices for the matched predicted mask, as related
# to the GT mask
# pred_match is an array of the indices linking predicted masks to a GT mask
# compare_matches also calculates overlaps following matching. overlaps.shape
# is [predicted_nuclei, gt_nuclei]
gt_match, pred_match, overlaps = utils.compute_matches(
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    iou_threshold=0.5, score_threshold=0.5)

# Use compute_matches to get IoU, recall, precision etc. 

#%% Calculating F1 score

# Total GT masks
ground_truths = gt_match.shape[0]

# Total identified masks
predictions = pred_match.shape[0]

# Max will return 0 if false positve is -ve. 
false_pos = max(0, predictions - ground_truths)



# Find the number of GT masks that have a predicted mask
# aka true positive
true_pos = np.where(gt_match > -1)[0].shape[0]

# Precision: (true positive)/(true positive + false positive)
precision = true_pos / (true_pos + false_pos)


# Find the indices of GT masks with no prediction
# aka false neg
false_neg = np.where(gt_match < 0)[0].shape[0]

# Recall: (true positive) / (true positive + false negative)
recall = true_pos / (true_pos + false_neg)

# F1: 2* (precision * recall) / (precision + recall)
f1 = 2 * (precision * recall) / (precision + recall)

# Mask rcnn stats:
mAP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
             r['rois'], r['class_ids'], r['scores'], r['masks'],
             iou_threshold=0.5)

# Total predictions that have a GT match / total number of predictions
np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)

#%% Return the IoU values

for i in overlaps:
    for j in i:
        if j != 0:
            print(j)
            
#%% Convert image dtype to uint8

out = np.zeros(image.shape, dtype=np.uint8)

out = cv2.normalize(image, out, 0, 255, cv2.NORM_MINMAX)

# Fixed! Converting the image to uint8 is the key! The strange gray 
# image wasn't because of RGB or anything, but instead due to the 
# metamorph image being uint16 rather than uint8

#%%


visualize.display_differences(
    out,
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    dataset.class_names, ax=get_ax(),
    show_box=False, show_mask=False,
    iou_threshold=0.5, score_threshold=0.5)












