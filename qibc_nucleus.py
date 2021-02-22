"""
Mask R-CNN

For training on the black+white image data from the Kaggle Data Science
Bowl 2018 dataset

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.io

# Load Mask RCNN
mrcnn_dir = os.path.abspath("Mask_RCNN")
sys.path.append(mrcnn_dir)
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize



class NucleusConfig(Config):
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
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
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


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    

class NucleiDataset(utils.Dataset):
    def load_nuclei(self, dataset_dir, subset=None):
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
        
        # Loading a training and validation subset of the data
        # Training is 90% of input data, validation is 10%
        #assert subset in ["train", "val"]
        
        if subset == "train":
            for i, img_id in enumerate(imgs):
                if i < int(len(imgs) * 0.9):
                    # For first 90% of the images, open as training
                    self.add_image(
                        "nucleus",
                        image_id=img_id,
                        path=os.path.join(dataset_dir, img_id, "images/{}.png".format(img_id))
                        )
        
        if subset == "val":
            for i, img_id in enumerate(imgs):
                if i >= int(len(imgs) * 0.9):
                    # For remaining 10% of the images, open for validation
                    self.add_image(
                        "nucleus",
                        image_id=img_id,
                        path=os.path.join(dataset_dir, img_id, "images/{}.png".format(img_id))
                        )
        else:
            for img_id in imgs:
                self.add_image(
                    "nucleus",
                    image_id=img_id,
                    path=os.path.join(dataset_dir, img_id, "images/{}.png".format(img_id))
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
    

