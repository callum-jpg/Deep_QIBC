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
from mrcnn import config


#%%

class AdjustNucleusConfigHigh(config.Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # Min max not available with pad64
        IMAGE_RESIZE_MODE = "pad64"
        IMAGE_MIN_SCALE = 2

config = AdjustNucleusConfigHigh()


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

#%% Exploring dataset

mask, class_ids = dataset.load_mask(image_id)

dataset.load_mask

test = dataset.load_image(image_id)

#%%  Load model

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

print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
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
utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                       r['rois'], r['class_ids'], r['scores'], r['masks'],
                       verbose=1)

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
gt_match, pred_match, overlaps = utils.compute_matches(
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    iou_threshold=0.5, score_threshold=0.5)

captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
    r['scores'][i],
    (overlaps[i, int(pred_match[i])]
        if pred_match[i] > -1 else overlaps[i].max()))
        for i in range(len(pred_match))]


#%%

test = skimage.io.imread(dataset.image_info[0]['path'])






