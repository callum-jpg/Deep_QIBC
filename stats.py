# Gathering useful stats to test model

#%% Calculating IoU

import tensorflow as tf

import qibc_nucleus
import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import time

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
import visualise



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
            # out = np.zeros(image.shape, dtype=np.uint8)
            # image = cv2.normalize(image, out, 0, 255, cv2.NORM_MINMAX)
                        
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
            
    def calculate_matches_and_f1(self, iou_level):
        """
        Find matches between gt masks and predicted masks. iou_level determines the
        minimum threshold to determine a match. 
        
        Returns a list of dicts that contains image, predicted masks, gt masks,
        their respective matches and f1 score.
        """
        
        for i, gt in enumerate(self.gt_detection_info):
        
            gt_match, pred_match, overlaps = utils.compute_matches(
                    gt["gt_bbox"], gt["gt_class_id"], gt["gt_mask"],
                    gt['rois'], gt['class_ids'], gt['scores'], gt['masks'],
                    iou_threshold=iou_level, score_threshold=0.5)
            
            # Add dict as an element in the gt_detection_info list
            self.gt_detection_info[i].update({"gt_match": gt_match,
                                              "pred_match": pred_match,
                                              "overlaps": overlaps,
                                              "f1": calculate_f1(gt_match, pred_match)[0],
                                              "precision": calculate_f1(gt_match, pred_match)[1],
                                              "recall": calculate_f1(gt_match, pred_match)[2]})
                       


def calculate_f1(gt_matches, pred_matches):
    """
    Calculates the F1, precision and recall score given matches gt and predicted mask from Mask R-CNN output
    """
    
    # For all of the predicted matches, find those that have a matched gt mask
    # and get the length of this array
    true_pos = np.where(pred_matches > -1)[0].shape[0]
    
    # len(pred_match) is all of the predicted masks, positive and negative
    precision = true_pos / len(pred_matches)

    # len(gt_match) is all of the true (+ve) and false (-ve) positives
    recall = true_pos / len(gt_matches)
        
    # Calculate f1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1, precision, recall
            
            
        
        
        
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
# Compare low/med/high config
# Zoomed in images, 


# Compare missed objects between detection methods


# New stats as above, but on precision/recall?


#%% low/med/high iou 0.7 f1 scores

# setting = ["low"]
setting = ["low", "med"]
# img_dir = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/u2os-20x"
img_dir = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/small_overlaps-BBBC004/test"
# img_dir = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/BBBC006-oof-u2os/images+masks/oof-images"


results = {"computation_setting": [],
           "precision": [],
           "recall": [],
           "f1_score": [],
           "execution_time": []}

for i in setting:
    start_time = time.time()
    stats = CalculateStats()
    stats.gt_detect(img_dir, i)
    stats.calculate_matches_and_f1(iou_level=0.7)
    
    res = stats.gt_detection_info[0]

    results["computation_setting"].append(i)
    results["precision"].append(stats.gt_detection_info[0]["precision"])
    results["recall"].append(stats.gt_detection_info[0]["recall"])
    results["f1_score"].append(stats.gt_detection_info[0]["f1"])
    results["execution_time"].append(time.time() - start_time)

    fig = visualise.display_detections(res)
    fig.savefig(i+" masks for small overlaps.png", dpi=300, bbox_inches='tight')
    
#%%
# Colours for plotting. From Solarized palette
colour_palette = [(42, 161, 152), (38, 139, 210), (108, 113, 196), (211, 54, 130)]
plot_colours = colour_palette
for i in range(len(colour_palette)):
	r, g, b = colour_palette[i]
	# Convert RGB (0, 255) to (0, 1) which matplotlib likes
	plot_colours[i] = (r / 255, g / 255, b / 255)
    
#%% IoU threshold F1 scores for small overlaps


# img_dir = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/u2os-20x"
img_dir = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/small_overlaps-BBBC004/test"
# img_dir = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/BBBC006-oof-u2os/images+masks/oof-images"

# Extract f1 scores for images over a range of IoU thresholds
# 0.1 to 0.9
iou_levels = np.round(np.arange(0.5, 0.95, 0.05), 2)



f1_scores = {"setting": []}
f1_scores.update({k: [] for k in iou_levels})


setting = ["low", "med"]
for i in setting:
    f1_scores["setting"].append(i)
    stats = CalculateStats()
    stats.gt_detect(img_dir, i)

    for iou_threshold in iou_levels:
        stats.calculate_matches_and_f1(iou_level=iou_threshold)
        
        for i, detection in enumerate(stats.gt_detection_info):
            f1_scores[iou_threshold].append(detection['f1'])
        


#%%

# f1 score vs IoU
# Compare low/med config
# i[1] to extract the second image f1_scores
y1 = [i[0] for i in list(f1_scores.values())[1:]]
y2 = [i[1] for i in list(f1_scores.values())[1:]]
# [1:] to ignore the first key, which is setting
x_values = list(f1_scores.keys())[1:]

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(x_values, y1, marker="o", color=plot_colours[3])
ax.plot(x_values, y2, marker="o", color=plot_colours[1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("IoU Threshold")
ax.set_ylabel("F1 Score")
ax.legend(["low", "med"])
fig.tight_layout()
fig.savefig("low vs med for small overlaps.png", dpi=300, bbox_inches='tight')
    
    

    
    



