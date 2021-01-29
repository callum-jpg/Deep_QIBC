import load_images
import detect

import numpy as np
from scipy import ndimage


class ProcessMasks(object):
    def __init__(self):
        self.masks = []
        self.labelled_masks = []
    
    def load_masks(self, data):
        """
        Extract the masks determined by Mask R-CNN.
        data: the output of DetectNuclei.run_detection(...)
        """
        
        for result in data:
            #print(result[1][0]['masks'])
            masks = {}
            
            masks.update({"image name": result[0],
                          "masks": result[1][0]['masks']})
            
            self.masks.append(masks)
        
        return self.masks
            
    def flatten_3d_to_2d(self, array):
        """
        'Flattens' a 3d array to 2d along k. For merging mask information
        found in k layers for an i, j dims image array
        """
        # Create an empty array of the input image shape
        flat_array = np.zeros(array.shape[0:2], dtype=np.int32)
        
        for i in range(0, array.shape[2]):
            # For each mask element, add to the 2d image array
            flat_array = flat_array + array[...,i]
            
        return flat_array
    
    def label_masks(self, array):
        """
        For an image array with i, j image dims and k masks, this function
        labels mask (k) elements individually.     
        """
        output_array = []
        
        for k_ind in range(0, array.shape[2]):
            
            # Label 
            loop_output, _ = ndimage.label(array[...,k_ind], np.ones((3, 3)))
            
            # Rename the identified labels with the corresponding mask number
            # Since masks are stored in different k elements, ndimage.label
            # only labels the mask as 1
            loop_output = np.where(loop_output==1, k_ind+1, loop_output)
            
            # Append arrays to list
            output_array.append(loop_output)
            
        # Stack list into 3d array
        output_array = np.stack(output_array, axis=2)
        
        return output_array
    
    
    def label_masks1(self, data):
        """
        Label masks and add to consolidated_mask list        
        """
        
        self.load_masks(data)
        
        for mask in self.masks:
            
            mask_labels = {}
            
            #print(masks1['masks'])
            
            labelled_masks = self.label_masks(mask['masks'])
            
            
            
            mask_labels.update({"image name": mask['image name'],
                                    "masks": self.flatten_3d_to_2d(labelled_masks)})
            
            
            self.labelled_masks.append(mask_labels)
        
        return self.labelled_masks
            
            
#%% Testing


# Load images
images = load_images.LoadImages()

images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

image_dir = "images"

images.load_images(image_dir)

# Run detection
nuclei_detection = detect.DetectNucleus()

# Select channel to run detection on (in this case, DAPI)
object_channel = images.channels[0]

nuclei_detection.run_detection(images.image_info, object_channel, "low", "cpu")

nuclei_detection.results

#%%
# Load and label masks
labelled = ProcessMasks()

labelled.label_masks1(nuclei_detection.results)

labelled.labelled_masks[0]['masks'].max()


#%%

labelled1 = ProcessMasks()

labelled1.load_masks(nuclei_detection.results)

for i in labelled1.masks:
    print(i)

                    