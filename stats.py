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

from detect import NUCLEUS_TRAINED_WEIGHTS, AdjustNucleusConfigLow, AdjustNucleusConfigMed, AdjustNucleusConfigHigh, LOGS_DIR
import load_images

#%%

class AdjustNucleusConfigHigh1(Config):
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


config = AdjustNucleusConfigHigh1()
#config = qibc_nucleus.NucleusInferenceConfig()

#%
# img_dir = 'datasets/nucleus/bw_train/'
img_dir = 'datasets/nucleus/label_test/'

dataset = qibc_nucleus.NucleiDataset()
dataset.load_nuclei(img_dir) 
dataset.prepare()

# image_id = random.choice(dataset.image_ids)
image_id = 1

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

image = cv2.normalize(image, out, 0, 255, cv2.NORM_MINMAX)
    
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

#%% Mask rcnn stats:
mAP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
             r['rois'], r['class_ids'], r['scores'], r['masks'],
             iou_threshold=0.5)

# Total predictions that have a GT match / total number of predictions
# (pred_match > -1) returns T/F depending if there is a match to a GT mask.
# This creates a bool array
# np.cumsum then calculates the cumulative sum where true = 1 and false = 0
# for the length of pred_match.
#
# (np.arange(len(pred_match)) + 1) then creates an array 
# (+ 1 starts it from 1 rather than 0) for the length of predicted matches
#
# Then, by dividing these it yields a iterative calculation of precision.
# The order of pred_match is from highest score to lowest (from compute_matches)
# np.cumsum when encountering false will not increase, thus by being divided
# by the steadily increasing np.arrange will lead to a value < 1.
# eg. 3 / 3 = 1, but 2 / 3 = 0.66...
# This solution from Mask R-CNN is beautiful and efficient

np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)

#%% Calculating F1

# For all of the predicted matches, find those that have a matched gt mask
# and get the length of this array
true_pos = np.where(pred_match > -1)[0].shape[0]

# len(pred_match) is all of the predicted masks, positive and negative
precision = true_pos / len(pred_match)

# len(gt_match) is all of the true (+ve) and false (-ve) positives
recall = true_pos / len(gt_match)

# Calculate f1
f1 = 2 * (precision * recall) / (precision + recall)


def calculate_f1(gt_matches, pred_matches):
    """
    Calculates the F1 score given matches gt and predicted mask from Mask R-CNN output
    """
    
    # For all of the predicted matches, find those that have a matched gt mask
    # and get the length of this array
    true_pos = np.where(pred_matches > -1)[0].shape[0]
    
    # len(pred_match) is all of the predicted masks, positive and negative
    precision = true_pos / len(pred_match)
    
    # len(gt_match) is all of the true (+ve) and false (-ve) positives
    recall = true_pos / len(gt_match)
    
    # Calculate f1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1
    
    
    
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

#%%

class CalculateStats:
    def __init__(self):
        # List to hold all of the gt images and masks
        self.gt_info = []
        self.gt_detection_info = []
        self.config = []

    def load_gt(self, img_dir, computation_requirement=None,):
        """
        Load images and their masks together. Rescale images to 8-bit.
        """        
        
        # Load optional configs
        if computation_requirement == "low":
            print("Using low settings")
            self.config = AdjustNucleusConfigLow()
        if computation_requirement == "med":
            print("Using medium settings")
            self.config = AdjustNucleusConfigMed()
            #self.config = AdjustNucleusConfigHigh1()
        if computation_requirement == "high":
            print("Using high settings")
            self.config = AdjustNucleusConfigHigh()
        # else:
        #     print("Using low settings")
        #     self.config = AdjustNucleusConfigLow()
    
        dataset = load_images.LoadImagesMasks()
        dataset.load_nuclei(img_dir) 
        dataset.prepare()
        
        for image_id in dataset.image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                        modellib.load_image_gt(dataset, self.config, image_id, use_mini_mask=False)
                        
            # Rescale to 8-bit
            # Since model was trained on 8-bit DSB18 images
            out = np.zeros(image.shape, dtype=np.uint8)
            image = cv2.normalize(image, out, 0, 255, cv2.NORM_MINMAX)
                        
            _gt_info = {"image": image,
                       "image_meta": image_meta,
                       "gt_class_id": gt_class_id,
                       "gt_bbox": gt_bbox,
                       "gt_mask": gt_mask}
        
            self.gt_info.append(_gt_info)
            
        
    def gt_detect(self, img_dir, computation_requirement=None, device=None):
        """
        Run detection on the images loaded by load_gt.
        
        Returns a list of dicts that contains image, predicted masks, gt masks
        """
        
        self.load_gt(img_dir, computation_requirement)
        
        # Load preferred device
        if device == "cpu":
            DEVICE = "/cpu:0"
        if device == "gpu":
            DEVICE = "/GPU:0"
        else:
            DEVICE = "/cpu:0"
            
        # Start model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=self.config)
            
        
        print("Loading weights ", NUCLEUS_TRAINED_WEIGHTS)
        model.load_weights(NUCLEUS_TRAINED_WEIGHTS, by_name=True)
        
        for gt in self.gt_info:
            #print(gt)
            results = model.detect_molded(np.expand_dims(gt["image"], 0), np.expand_dims(gt["image_meta"], 0), verbose=1)
            
            # Merge GT info and results into one dict
            gt_and_detection = {**gt, **results[0]}
            
            self.gt_detection_info.append(gt_and_detection)
            
    def calculate_matches_and_f1(self, iou):
        """
        Find matches between gt masks and predicted masks. iou determines the
        minimum threshold to determine a match. 
        
        Returns a list of dicts that contains image, predicted masks, gt masks,
        their respective matches and f1 score.
        """
        
        for i, gt in enumerate(self.gt_detection_info):
        
            gt_match, pred_match, overlaps = utils.compute_matches(
                    gt["gt_bbox"], gt["gt_class_id"], gt["gt_mask"],
                    gt['rois'], gt['class_ids'], gt['scores'], gt['masks'],
                    iou_threshold=iou, score_threshold=0.5)
            
            # Add dict as an element in the gt_detection_info list
            self.gt_detection_info[i].update({"gt_match": gt_match,
                                              "pred_match": pred_match,
                                              "overlaps": overlaps,
                                              "f1": calculate_f1(gt_match, pred_match)})
                       


def calculate_f1(gt_matches, pred_matches):
    """
    Calculates the F1 score given matches gt and predicted mask from Mask R-CNN output
    """
    
    # For all of the predicted matches, find those that have a matched gt mask
    # and get the length of this array
    true_pos = np.where(pred_matches > -1)[0].shape[0]
    
    # len(pred_match) is all of the predicted masks, positive and negative
    precision = true_pos / len(pred_matches)

    # len(gt_match) is all of the true (+ve) and false (-ve) positives
    recall = true_pos / len(gt_matches)
    print("Precision: {}. Recall: {}".format(precision, recall))    
    # Calculate f1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1
            
            
        
        
        
#%%

test = CalculateStats()

test.gt_detect('datasets/nucleus/label_test/', computation_requirement="low")

#%%

test.calculate_matches_and_f1(iou=0.1)

test.gt_detection_info[1]['f1']

#%% Extract IoU values for individual nuclei

overlaps = test.gt_detection_info[0]['overlaps']

overlaps[np.where(overlaps!=0)]

#%% Generating detection stats

#np.linspace(0.5, 1, num=6)

# Extract f1 scores for images over a range of IoU thresholds
# 0.1 to 0.9
iou_levels = np.round(np.arange(0.1, 0.95, 0.05), 2)

f1_scores = {}
for iou_threshold in iou_levels:
    f1_scores.update({"image_number": []})
    
    iou_key = ("f1_"+str(iou_threshold))
    f1_scores.update({iou_key: []})
    
    test.calculate_matches_and_f1(iou=iou_threshold)
    
    for i, detection in enumerate(test.gt_detection_info):
        f1_scores["image_number"].append(i + 1)
        f1_scores[iou_key].append(detection['f1'])

#%%

# f1 score vs IoU
# Compare low/med config
# i[1] to extract the second image f1_scores
y_values = [i[0] for i in list(f1_scores.values())[1:]]
# [1:] to ignore the first key, which is image_number
x_values = list(f1_scores.keys())[1:]

fig, ax = plt.subplots(1, 1)
ax.plot(x_values, y_values, marker="o")
ax.set_xticklabels(labels=x_values, rotation=45, ha="right", rotation_mode="anchor")



# Dot bar plot of f1 IoU for multiple images
# Compare low/med config
# Zoomed in images, 





# Compare missed objects between detection methods


# New stats as above, but on precision/recall?


#%%


    
    



