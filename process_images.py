import load_images
import detect
import skimage.color
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
            # print(result[1][0]['masks'])
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
            flat_array = flat_array + array[..., i]

        return flat_array

    def label_masks_and_edges(self, array):
        """
        For an image array with i, j image dims and k masks, this function
        labels mask (k) elements individually.
        """
        output_masks = []
        output_edges = []

        for k_ind in range(0, array.shape[2]):

            # Label
            loop_output_masks, _ = ndimage.label(array[..., k_ind], np.ones((3, 3)))
            
            loop_output_edges = skimage.segmentation.find_boundaries(array[..., k_ind], 
                                                                      mode="inner", background=0).astype(np.uint8)
            
            # loop_output_edges = ndimage.label(array[..., k_ind], np.ones((3, 3)))


            # Rename the identified labels with the corresponding mask number
            # Since masks are stored in different k elements, ndimage.label
            # only labels the mask as 1
            loop_output_masks = np.where(loop_output_masks == 1, k_ind + 1, loop_output_masks)
            
            loop_output_edges = np.where(loop_output_edges == 1, k_ind + 1, loop_output_edges)

            # Append arrays to list
            output_masks.append(loop_output_masks)
            output_edges.append(loop_output_edges)

        # Stack list into 3d array
        output_masks = np.stack(output_masks, axis=2)
        output_edges = np.stack(output_edges, axis=2)

        flat_ouput_masks = self.flatten_3d_to_2d(output_masks)
        
        flat_output_edges = self.flatten_3d_to_2d(output_edges)

        return flat_ouput_masks, flat_output_edges

    def label_masks(self, data):
        """
        Label masks and add to consolidated_mask list.
        """

        self.load_masks(data)

        for mask in self.masks:

            mask_labels = {}

            labelled_masks, labelled_edges = self.label_masks_and_edges(mask['masks'])

            mask_labels.update({"image name": mask['image name'],
                                "masks": labelled_masks,
                                "edges": labelled_edges
                                })

            self.labelled_masks.append(mask_labels)

        return self.labelled_masks

class RecordIntensity(object):
    def __init__(self, image_info, labelled_masks, channels=None):
        self.x = []
        self.image_info = image_info
        self.masks = labelled_masks
        if channels is None:
            self.channels = ["image_data"]
        else:
            self.channels = channels
            
        #print(self.masks)
        
    def consolidate_masks_with_images(self, object_channel=None):
        """
        For masks identified by Mask R-CNN and labelled, consolidate
        these masks with the appropriate image set based on filename.
        
        The image channel used to determine objects will be used for matching.
        
        TODO: Currently, this works by matching stage position. This method
        wont work for images with different filename structures. Add a REGEX
        grouping option into the load_images module that can reliably group
        images based on user preference. 
        """
        
        if object_channel is None:
            print("No object channel provided. Continuing with default.")
            object_channel = "image_data"
        
        for img in self.image_info:
            for mask in self.masks:
                #print("this is a mask from self.masks:", mask)
                
                # print(img[object_channel][0])
                #print(img[object_channel][0])
                # Check that the filenames match
                if img[object_channel][0] == mask['image name'][0]:
                    print("hit", mask['image name'][0])
                    img.update({'masks': mask['masks'],
                                'edges': mask['edges']})

    def record_intensity(self):
        """
        Record the intensity of the image within the given object mask. 
        Labels are defined the mask
        
        img is a 2d array of a multichannel image
        """
        
        output_data = {'image number': [],
                       'object number': [],
                       'object_area': []}
        
        # Build dictionary to store intensity values for each channel
        # Entire nuclei intensities
        output_data.update(("".join((channel, "_mean_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_total_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_min_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_max_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_std_intensity")), []) for channel in self.channels)
        
        # Edge nuclei intensities
        output_data.update(("".join((channel, "_edge_mean_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_edge_total_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_edge_min_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_edge_max_intensity")), []) for channel in self.channels)
        output_data.update(("".join((channel, "_edge_std_intensity")), []) for channel in self.channels)
        
        for img in self.image_info:
            masks = img['masks']
            # Find the indices of non.zero numbers and use these
            # indices to extract out non.zero values from original array
            mask_min = np.min(masks[np.nonzero(masks)])
            mask_max = np.max(masks[np.nonzero(masks)])
            object_number = [obj for obj in range(mask_min, mask_max + 1)]                 
            output_data['object number'] = output_data['object number'] + object_number
            output_data['image number'] = output_data['image number'] + [img['image number']] * mask_max
            # np.bincount returns the bincount for ints found in the inputted array.
            # In this case, an array of either 0's or 1/2/3/etc. is given. Thus, 
            # np.bincount counts the number of 0's or the number of 2's, for example. [1:]
            # slices out the count for the latter, rather than the 0's count. 
            # The final [0] slices out this value from the np.array
            area = [np.bincount(np.equal(masks, obj).flat)[1:][0] for obj in object_number]
            output_data['object_area'] = output_data['object_area'] + area
            
            for channel in self.channels:
                # Convert image array to grayscale
                img_gray = skimage.color.rgb2gray(img[channel][1])
                
                ## Entire object intensities
                # Mean intensity
                mean_intensity = [ndimage.mean(img_gray, 
                                                  labels=(np.equal(img['masks'], obj)))
                                                      for obj in object_number]
                # Add recorded intensities to corresponding channel dict key
                output_data[channel+"_mean_intensity"] = output_data[channel+"_mean_intensity"] + mean_intensity                
        
                # Measure total intensity
                total_intensity = [ndimage.sum(img_gray, 
                                                  labels=(np.equal(img['masks'], obj)))
                                                      for obj in object_number]
                output_data[channel+"_total_intensity"] = output_data[channel+"_total_intensity"] + total_intensity   
                
                # Min intensity
                min_intensity = [ndimage.minimum(img_gray, 
                                                  labels=(np.equal(img['masks'], obj)))
                                                      for obj in object_number]
                output_data[channel+"_min_intensity"] = output_data[channel+"_min_intensity"] + min_intensity  
                
                # Max intensity
                max_intensity = [ndimage.maximum(img_gray, 
                                                  labels=(np.equal(img['masks'], obj)))
                                                      for obj in object_number]
                output_data[channel+"_max_intensity"] = output_data[channel+"_max_intensity"] + max_intensity  
                
                # Standard deviation
                std_intensity = [np.sqrt(
                    ndimage.mean((img_gray - mean_intensity[obj - 1]) ** 2, 
                                                  labels=(np.equal(img['masks'], obj))))
                                                      for obj in object_number]
                output_data[channel+"_std_intensity"] = output_data[channel+"_std_intensity"] + std_intensity   
                
                ## Edge intensities
                # Mean edge intensity
                mean_intensity_edge = [ndimage.mean(img_gray, 
                                                    labels=(np.equal(img['edges'], obj))) 
                                       for obj in object_number]
                output_data[channel+"_edge_mean_intensity"] = output_data[channel+"_edge_mean_intensity"] + mean_intensity_edge      
                
                # Total edge intensity
                total_intensity_edge = [ndimage.sum(img_gray, 
                                                  labels=(np.equal(img['edges'], obj)))
                                                      for obj in object_number]
                # Add recorded intensities to corresponding channel dict key
                output_data[channel+"_edge_total_intensity"] = output_data[channel+"_edge_total_intensity"] + total_intensity_edge  
        
                # edge min intensity
                min_intensity_edge = [ndimage.minimum(img_gray, 
                                                  labels=(np.equal(img['edges'], obj)))
                                                      for obj in object_number]
                output_data[channel+"_edge_min_intensity"] = output_data[channel+"_edge_min_intensity"] + min_intensity_edge  
                
                # edge max intensity
                max_intensity_edge = [ndimage.maximum(img_gray, 
                                                  labels=(np.equal(img['edges'], obj)))
                                                      for obj in object_number]
                output_data[channel+"_edge_max_intensity"] = output_data[channel+"_edge_max_intensity"] + max_intensity_edge                  
        
                # Standard deviation of edge intensity
                std_intensity_edge = [np.sqrt(
                    ndimage.mean((img_gray - mean_intensity[obj - 1]) ** 2, 
                                                  labels=(np.equal(img['edges'], obj))))
                                                      for obj in object_number]
                output_data[channel+"_edge_std_intensity"] = output_data[channel+"_edge_std_intensity"] + std_intensity_edge   
        
        return output_data

            
        
# #%% Testing


# # Load images
# images = load_images.LoadImages()

# images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

# image_dir = "images"

# images.load_images(image_dir)

    
# #%%
# # Run detection
# nuclei_detection = detect.DetectNucleus()

# # Select channel to run detection on (in this case, DAPI)
# object_channel = images.channels[0]

# nuclei_detection.run_detection(images.image_info, "low", "cpu", object_channel)

# nuclei_detection.results


# #%%
# # Load and label masks
# labelled = ProcessMasks()

# labelled.label_masks(nuclei_detection.results)

# labelled.labelled_masks[0]['masks'].max()

# #%%

# intensity = RecordIntensity(images.image_info, labelled.labelled_masks, ["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

# # Add itentified masks to image_info
# intensity.consolidate_masks_with_images("w1DAPI")

# data = intensity.record_intensity()

# #%%

# for i in intensity.masks:
#     print(i['masks'])

# #%%

# intensity = RecordIntensity()

# data = intensity.record_intensity()

# #%%

# import pandas as pd

# df = pd.DataFrame(data=data)

# df.to_csv('test.csv', index=False)

# #%%

# channels = images.channels

# d = dict(((channel, 'hello'), []) for channel in channels)

# da = {}

# da.update(((channel, 'hello'), []) for channel in channels)

# d.update({'test': 0})

# d.update({channels[0]: [1, 2, 3]})

# d[channels[0]] + [4, 5, 6]

# d[channels[0]] = d[channels[0]] + [4, 5, 6]

# #%%

# for channel in channels:
#     print(channel, 'hello')


# #%%



# from difflib import get_close_matches, SequenceMatcher

# # From image_info access specific filename for a given channel
# img_str = images.image_info[0]['w1DAPI'][0]

# img_str1 = images.image_info[0]['w2Cy5'][0]

# img_str2 = images.image_info[1]['w2Cy5'][0]

# # For object masks, access filename with 
# mask_str = labelled.labelled_masks[0]['image name']


# get_close_matches(word, possibilities)


