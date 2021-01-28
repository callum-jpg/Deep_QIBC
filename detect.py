import sys
import os
import tensorflow as tf

mrcnn_dir = os.path.abspath("Mask_RCNN")
sys.path.append(mrcnn_dir)
import mrcnn.model as modellib

# Nucleus directory
nuc_dir = os.path.abspath("Mask_RCNN/samples/nucleus")
sys.path.append(nuc_dir)
import nucleus

# Path to nucleus trained weights
NUCLEUS_TRAINED_WEIGHTS = 'weights/mask_rcnn_nucleus2.h5'

# Logs dir
LOGS_DIR = "logs"


class AdjustNucleusConfig(nucleus.NucleusInferenceConfig):
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
    

class DetectNucleus:
    def __init__(self):
        self.empty = []
        self.config = AdjustNucleusConfig()
        
    def load_config(self, computation_requirement):
        """
        Load the nucleus config from Mask R-CNN with an option to alter
        the computation requirement of the model
        
        A lower computation requiremnet will yield lower accuracy but increased
        analysis speed.
        """

        

#%% testing

DEVICE = "/cpu:0" 

config = DetectNucleus().config

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)


        
        