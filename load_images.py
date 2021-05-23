import os
from difflib import get_close_matches
import skimage.io
import re
import numpy as np
import cv2
import sys

mrcnn_dir = os.path.abspath("Mask_RCNN")
sys.path.append(mrcnn_dir)
from mrcnn import utils

class LoadImages:
    def __init__(self, grouping_regex):
        self.image_info = []
        self.channels = []
        self.grouping_regex = grouping_regex
        
        
    def add_channels(self, channels):
        """
        Add strings which correspond to image channels. Will be used to delineate
        filenames
        """
        for channel in channels:
            self.channels.append(channel)
        
        return self.channels
    
    
    def match_images1(self, image_path):
        """
        Performs a fuzzy string match on a filename list.
        Strings are matched into lists with the same length as self.channels.
        Removes channel strings from filename to aid in matching.
        
        This method doesn't work well and was replaced with match_images that 
        relies on custom regex.
        """
        
        # Set path in which images are found
        self.image_path = image_path
        
        filelist = os.listdir(image_path)
        
        # define non_grouping element of filename - eg channel
        non_group = self.channels
        
        # List of lists for similarly named images (eg stage pos)
        self.grouped_images = []
    
        # Remove non_group from elements of list
        strip_list = [file.replace(ng, '') for ng in non_group for file in filelist if ng in file]
    
        # Identify unique elements
        unique = list(set(strip_list))
        # Order with the end of the string (stage number) and sort
        unique = sorted(unique, reverse=True, key=lambda x: x[::-1])
    
        # Iterate through unique elements and find fuzzy matches in filelist
        for i in unique:
            sub_list = get_close_matches(i, filelist, n=len(self.channels), cutoff=0.9)
            self.grouped_images.append(sub_list)   
            
        # Make sure it returns an equal number of elements to the original filelist
        if len(filelist) != sum(len(x) for x in self.grouped_images):
            #return "Images inaccurately grouped"
            raise ValueError("Images unable to be accurately grouped")
        else:
            print (self.grouped_images)
            return self.grouped_images
        
    def match_images(self, image_path, regex):
        """
        Matching based on a given regex
        """
        
        filelist = os.listdir(image_path)
        
        unique = []
        for i in filelist:
            unique += re.findall(regex, i)
        # Remove duplicates
        unique = set(unique)
        
        self.grouped_images = []
        # Iterate through unique identifiers,
        # Find index of files which contain this identifer,
        # Add these indexed filenames to sub_group as a list
        # Append this sub_group list to grouped_images
        for image_set in unique:
            file_indices = [i for i, s in enumerate(filelist) if image_set in s]
            sub_group = []
            for idx in file_indices:
                sub_group.append(filelist[idx])
            self.grouped_images.append(sub_group)
        return self.grouped_images        
        

    def add_images(self):
        """
        Add grouped images to dict
        """

        def sort_images(image_group):
            """
            Returns a dictionary that arranges image names in the dict values
            with the corresponding channel as the key.
            
            Image names are the first element in the list for a given key.
            
            Function is called in the following for loop. image_group is
            the element of grouped_images. 
            """
            # I don't like this. I think it's inaccurate and clunky
            output_dict = {}
            for channel_img in image_group:
                if len(self.channels) == 0:
                    # No channel names given. Load single image
                    output_dict.update({'image_data': [channel_img]})
                    # If channels are 0, objects will be detected in all images
                    output_dict.update({'object_image': [channel_img]})
                else: 
                    for channel in self.channels:
                        if channel in channel_img:
                            # Values in list so read img array can be appended
                            output_dict.update({channel: [channel_img]})
                    if self.object_channel in channel_img:
                        output_dict.update({"object_image": [channel_img]})
                        # print(output_dict["object_image"])
            
            return output_dict
        
        # Iterate over all image groups and build a dict that contains
        # image number, path, and each of the image channels with associated
        # filename for a given image
        for image_id, image_set in enumerate(self.grouped_images):
            image_info = {}
            #print(image_set)
            image_info.update({"image number": image_id+1, # Start at 1
                               "path": self.image_path})
            
            image_info.update(sort_images(image_set))
            
            self.image_info.append(image_info)
            
    def load_images(self, image_path, object_channel=None):
        """
        Loads 2d array (grayscale) for each image into image_info.
        image_info = [{
            "image number": "",
            "path": "",
            "channel1+n": ["image_filename", "image_array"],
            }]
        """
        
        # Set path in which images are found
        self.image_path = image_path
        
        self.object_channel = object_channel
        
        # Perform grouping of images into image sets
        # If there are 2+ channels, perform grouping of image sets into a
        # nested list
        if len(self.channels) > 1:
            self.match_images(image_path, self.grouping_regex)
        else:
            self.grouped_images = [[i] for i in os.listdir(image_path)]
            
        
        # Build image_info dict
        self.add_images()
        
        # Load image information as np arrays to corresponding channel
        for image_id, image_set in enumerate(self.image_info):
            #print(image_id, image_set)
            
            if len(self.channels) == 0:
                open_path = os.path.join(image_set["path"], image_set["image_data"][0])
                # _open_image = skimage.io.imread(open_path)
                # skimage.io.imread opens image as dtype uint16. While this
                # works for nuclei detection, later visualisation of images 
                # with matplotlib imshow requires unit8. img_as_ubyte converts
                # to unit8 (will need further conversion from RGB to gray, though)
                # open_image = skimage.img_as_ubyte(_open_image)
                

                # Image opened 'as is' and will be used for intensity readings. 
                open_image = skimage.io.imread(open_path)
                
                # Mask R-CNN trained on 8-bit images but converting input images
                # to 8-bit will lead to loss in precision for intensity readings.
                # Instead, create a new 8-bit image that will be used only for detection
                object_image = skimage.img_as_ubyte(open_image)
                
                # # Rescale to 8-bit
                # # Since model was trained on 8-bit DSB18 images
                # out = np.zeros(open_image.shape, dtype=np.uint8)
                # open_image = cv2.normalize(open_image, out, 0, 255, cv2.NORM_MINMAX)         
                
                if object_image.ndim != 3: # Convert image to RGB if not already
                    object_image = skimage.color.gray2rgb(object_image)
                self.image_info[image_id]["image_data"].append(open_image)
                self.image_info[image_id]["object_image"].append(open_image)
                
                
            else:
                for channel in self.channels:
                    if object_channel == None:
                        raise Exception("No object channel selected for a muilti channel image")
                    
                    open_path = os.path.join(image_set["path"], image_set[channel][0])
                    # _open_image = skimage.io.imread(open_path)
                    # skimage.io.imread opens image as dtype uint16. While this
                    # works for nuclei detection, later visualisation of images 
                    # with matplotlib imshow requires unit8. img_as_ubyte converts
                    # to unit8 (will need further conversion from RGB to gray, though)
                    #open_image = skimage.img_as_ubyte(_open_image)
                    
                    if object_channel != None:
                        _open_image = skimage.io.imread(open_path)
                        object_image = skimage.img_as_ubyte(_open_image)
                        if object_image.ndim != 3: # Convert image to RGB if not already
                            object_image = skimage.color.gray2rgb(object_image)
                        self.image_info[image_id]["object_image"].append(object_image)

                    open_image = skimage.io.imread(open_path)           
                        
                    self.image_info[image_id][channel].append(open_image)

        return self.image_info


class LoadImagesMasks(utils.Dataset):

    def load_nuclei(self, dataset_dir):
        """
        Load images containing nuclei for model testing
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

        for img_id in imgs:
            #print(img_id)
            self.add_image(
                "nucleus",
                image_id=img_id,
                path=os.path.join(dataset_dir, img_id, "images/{}.png".format(img_id))
                )

    def load_mask(self, image_id):
        """
        Load masks for the corresponding nuclei images
        """
        
        info = self.image_info[image_id]
        
        # Specifically redirect to the masks subfolder
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        
        mask = []
        # Open all masks.png images in mask directory and store in array
        for file in next(os.walk(mask_dir))[2]:
            if file.endswith(".png"):
                # open and read the mask as a bool array
                # as_gray to make sure it's a 1 channel image
                individual_mask = skimage.io.imread(os.path.join(mask_dir, file), as_gray=True).astype(np.bool)
                # Add the opened mask to the mask list
                mask.append(individual_mask)
        
        # Stack joins arrays along a new axis. Different to concatenate which joins
        # them on the same axis
        mask = np.stack(mask, axis=-1)

        # Return the list of mask arrays and an array of class IDs. These are,
        # all 1's for however many masks were identified. [-1] selects the last
        # number in the shape 
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    

        
                
                
                
    
#%% Testing

# images = LoadImages()

# images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

# image_dir = "images"

# images.load_images(image_dir)

# #%% Get size of image_info in bytes

# from sys import getsizeof

# #print(getsizeof(images.image_info))
