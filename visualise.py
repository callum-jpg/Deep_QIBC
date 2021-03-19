# Aim: to make simple, intuitive plots to reveal how the predictions are doing

import matplotlib.pyplot as plt
import process_images
import colorsys
import numpy as np


#%%

def display_detections(detection_output):
    """
    Displays the input image and predicted masks.
    
    If detection_output is from gt_detect, it will display the associated f1
    score, if present
    """
    
    input_image = detection_output["image"].copy()
    pred_masks = detection_output["masks"].copy()
    gt_masks = detection_output["gt_mask"].copy()
    
    # Display stats?
    if "f10.7" in detection_output.keys():
        plot_stats = True
    else: 
        plot_stats = False
    
    fig, ax = plt.subplots(1, 3 if plot_stats else 2, figsize=(15, 20))
    
    ax[0].imshow(input_image)
    ax[0].axis('off')
    ax[0].set_title("Input image")
    
    ax[1].imshow(colour_masks(input_image, pred_masks))
    ax[1].axis('off')
    ax[1].set_title("Predicted masks")
    
    if plot_stats:
        ax[2].imshow(display_difference(input_image, gt_masks, pred_masks))
        ax[2].axis("off")
        # I don't like how this triple quote looks, but it's how to make it
        # align above the plot correctly
        ax[2].set_title("""
Ground truth vs. predicted masks
Green = True positve, red = True negative, 
blue = False positive
F1@0.7 IoU score: {}""".format(round(detection_output["f10.7"], 3)))
    fig.tight_layout()
    
    return fig
    
#display_detections(res) 
#fig.savefig("test.png", dpi=300, bbox_inches='tight')   
    


#%%
def generate_colours(num_colours):
    """
    Taken from Mask RCNN
    
    Creates visually distinct colours by first generating them in HSV
    before converting them to RGB
    """    
    # Create a list of tuples with iterative hsv values
    # hue: cyclinder of colours, 0-1 (0-360). Segment this into number of cols req.
    # saturation: always 1
    # brightness: always 1
    hsv = [(i / num_colours, 1, 1) for i in range(num_colours)]
    
    # Convert list of tuples into list of RGB value tuples
    # Map applies the lambda function across the hsv list of tuples
    # Map is applied to each element of the list, ie each tuple
    # * unpacks iterables in each of the tuples
    colours = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
    
    return colours

def apply_mask(image, mask, colour):
    """
    Taken from Mask RCNN

    For a given mask, apply this to the given image
    """
    # Alpha for the masks
    alpha = 0.5
    
    
    # The input image for Mask RCNN is rgb, hence image.shape[-1] == 3
    # Moreover, the colour argument represents colours in rgb mode.
    for channel in range(3):
        # The mask array returns 1 where a predicted mask lays
        # np.where finds where the a mask is present (ie 1)
        # It then applies a transformation to the indices where the mask occurs
        # ie image[:,:,channel] * (1 - alpha) + alpha * colour[channel] * 255
        # otherwise pixels reamins as image[:,:,channel].
        # In this instance, np.where basically converts the mask array
        # into an array of identical size 
        image[:,:,channel] = np.where(mask == 1,
                                image[:,:,channel] * (1 - alpha) +
                                alpha * colour[channel] * 255,
                                image[:,:,channel])
        
    return image

def colour_masks(image, pred_masks):
    """
    Generates an image with coloured masks applied
    """
    
    mask_count = pred_masks.shape[-1]
    
    # Generate unique RGB colours for each mask
    colours = generate_colours(mask_count)
    
    # Convert image ready to be masked
    masked_img = image.astype(np.uint32).copy()
    
    for i in range(mask_count):
        i_mask = pred_masks[:,:,i]
        
        masked_img = apply_mask(image, i_mask, colours[i])
        
    #return masked_img
    return image
    
# test = colour_masks(res["image"], res["masks"])

# fig, ax = plt.subplots(1, 1)
# ax.imshow(test)

def display_difference(image, gt_masks, pred_masks):
    """
    Return an image that shows areas predicted corrently (green) and areas
    missed (red)
    """
    
    # Create an empty array, ready to be coloured, with input image shape
    img = np.zeros(image.shape, dtype=np.uint8)

    # Red will mark unmatched GT and green will match where GT overlaps
    # with prediction
    # Blue will mark areas that were predicted, but are not present in the 
    # GT 
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    # Flatten masks into 2d array depicting area which contains a mask
    # Resolution of overlapping masks is lost    
    gt = flatten_masks(gt_masks)
    pred = flatten_masks(pred_masks)
    
    # True positive overlaps
    # Green
    tp_overlaps = np.where((gt == 1) & (pred == 1), 1, np.zeros(gt.shape))
    
    # False negative overlaps
    # Red
    fn_overlaps = np.where((gt == 1) & (pred == 0), 1, np.zeros(gt.shape))
    
    # False positive overlaps
    # Blue
    fp_overlaps = np.where((gt == 0) & (pred == 1), 1, np.zeros(gt.shape))
    
    for channel in range(3):
        img[:,:,channel] = np.where(tp_overlaps == 1, green[channel], img[:,:,channel])
        img[:,:,channel] = np.where(fn_overlaps == 1, red[channel], img[:,:,channel])
        img[:,:,channel] = np.where(fp_overlaps == 1, blue[channel], img[:,:,channel])
    
    return img

# fig, ax = plt.subplots(1, 1)            
# ax.imshow(display_difference1(res["image"], res["gt_mask"], res["masks"]))


def flatten_masks(array):
    """
    'Flattens' a 3d array to 2d along k. For merging mask information
    found in k layers for an i, j dims image array.
    
    This function also ignores any overlapping objects
    """
    # Create an empty array of the input image shape
    flat_array = np.zeros(array.shape[0:2], dtype=np.int32)

    for i in range(0, array.shape[2]):
        # For each mask element, add to the 2d image array
        flat_array = flat_array + array[..., i]
    
    # Overlaps will return True + True = 2 in the above loop and np.where
    # equates them to 1. Thus, resolution of overlapping masks is lost 
    # by this function. 
    flat_array = np.where(flat_array > 1, 1, flat_array)

    return flat_array


#%% 

def display_difference1(image, gt_masks, pred_masks):
    """
    Return an image that shows areas predicted corrently (green) and areas
    missed (red)
    
    Old function, doesn't colour accurately:
    # When iterating through the channels and checking if the 
    # pixel looks like red[c], technically ALL pixels look like
    # red (thus True) when in channels 1:2.
    # This leads to pixels being coloured green despite no
    # gt_mask match at that position. This is also why some
    # pixels return yellow. Find a better way of matching 
    # pred to gt rather than just rgb colours. 
    """
    
    # Create an empty array, ready to be coloured, with input image shape
    img = np.zeros(image.shape, dtype=np.uint8)
    
    # Extract the loaded GT masks and pred masks
    # gt_masks = detection_results["gt_mask"]
    # pred_masks = detection_results["masks"]
    
    # Red will mark unmatched GT and green will match overlapping
    # gt with pred
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    

    
    for i in range(gt_masks.shape[-1]):
        for c in range(3): # 3 channel RGB
            # Select the mask to apply to image
            gt_mask = gt_masks[:,:,int(i)]
            # For each channel, colour green where a gt mask is found
            img[:,:,c] = np.where(gt_mask == 1,
                                  img[:,:,c] + red[c],
                                  img[:,:,c])
    
    # Now, to colour green where there is a predicted mask
    for i in range(pred_masks.shape[-1]):   
        for c in range(3):
            pred_mask = pred_masks[:,:,int(i)]
            
            # Where the pixel is a pred_mask AND it has been coloured red
            # (therefore a gt_mask was marked there).
            # Add the green colour and remove red markings
            # If you don't remove red, overlaps become yellow.
            img[:,:,c] = np.where((pred_mask == 1) & (img[:,:,c] == red[c]),
                                      img[:,:,c] + green[c] - red[c],
                                      img[:,:,c])
            # PROBLEM WITH ABOVE:
                # When iterating through the channels and checking if the 
                # pixel looks like red[c], technically ALL pixels look like
                # red (thus True) when in channels 1:2.
                # This leads to pixels being coloured green despite no
                # gt_mask match at that position. This is also why some
                # pixels return yellow. Find a better way of matching 
                # pred to gt rather than just rgb colours. 
            
            
    # # Now, colour pixels blue where there is a pred_match but no gt_match
    # for i in range(pred_masks.shape[-1]):   
    #     for c in range(3):
    #         pred_mask = pred_masks[:,:,int(i)]
            
    #         # Where the pixel is a pred_mask AND it has been coloured red
    #         # (therefore a gt_mask was marked there).
    #         # Add the green colour and remove red markings
    #         # If you don't remove red, overlaps become yellow.
    #         img[:,:,c] = np.where((pred_mask == 1) & (img[:,:,c] != red[c]) & (img[:,:,c] != green[c]),
    #                                   img[:,:,c] + blue[c] - green[c],
    #                                   img[:,:,c])
    
    return img
        
    
# fig, ax = plt.subplots(1, 1)            
# ax.imshow(display_difference(res["image"], res["gt_mask"], res["masks"]))


