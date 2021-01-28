import os
from difflib import get_close_matches
import skimage.io

class LoadImages:
    def __init__(self):
        self.image_info = []
        self.channels = []
        
        
    def add_channels(self, channels):
        """
        Add strings which correspond to image channels. Will be used to delineate
        filenames
        """
        for channel in channels:
            self.channels.append(channel)
        
        return self.channels
    
    
    def match_images(self, image_path):
        """
        Performs a fuzzy string match on a filename list.
        Strings are matched into lists with the same length as self.channels.
        Removes channel strings from filename to aid in matching.
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
            return self.grouped_images

    def add_images(self):
        """
        Add grouped images to dict
        """
        
        def sort_images(image_set):
            """
            Arranges images as dict values with the corresponding channel as the key
            """
            output_dict = {}
            for channel_img in image_set:
                for channel in self.channels:
                    if channel in channel_img:
                        # Values in list so read img array can be appended
                        output_dict.update({channel: [channel_img]})
            return output_dict
        
        for image_id, image_set in enumerate(self.grouped_images):
            image_info = {}
            #print(image_set)
            image_info.update({"image number": image_id+1,
                               "path": self.image_path}) # Start at 1
            image_info.update(sort_images(image_set))
            self.image_info.append(image_info)
            
    def load_images(self, image_path):
        """
        Loads 2d array (grayscale) for each image into image_info.
        image_info = [{
            "image number": "",
            "path": "",
            "channel1+n": ["image_filename", "image_array"],
            }]
        """
        
        # Perform grouping of images into image sets
        self.match_images(image_path)
        
        # Build image_info dict
        self.add_images()
        
        # Load image information as np arrays to corresponding channel
        for image_id, image_set in enumerate(self.image_info):
            for channel in self.channels:
                open_path = os.path.join(image_set["path"], image_set[channel][0])
                _open_image = skimage.io.imread(open_path)
                # skimage.io.imread opens image as dtype uint16. While this
                # works for nuclei detection, later visualisation of images 
                # with matplotlib imshow requires unit8. img_as_ubyte converts
                # to unit8 (will need further conversion from RGB to gray, though)
                open_image = skimage.img_as_ubyte(_open_image)                
                if open_image.ndim != 3: # Convert image to RGB if not already
                    open_image = skimage.color.gray2rgb(open_image)
                self.image_info[image_id][channel].append(open_image)
                
                
                
    
#%% Testing

images = LoadImages()

images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

image_dir = "images"

images.load_images(image_dir)

#%% Get size of image_info in bytes

from sys import getsizeof

#print(getsizeof(images.image_info))
