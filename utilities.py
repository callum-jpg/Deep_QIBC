import os
import skimage.io
import skimage.draw
import json
import numpy as np 
import shutil
import skimage.morphology

#%%

def json_to_png_mask(gt_image, json_masks):
    """
    Converts masks as defined by polygons in a JSON into individual png masks.
    
    Input GT image and generated masks are saved into image and masks folders,
    respectively.
    """
    
    img = skimage.io.imread(gt_image, as_gray=True)
    height, width = img.shape
    
    # Get the filename and remove the file extension
    filename = os.path.basename(gt_image).split(".")[0]
    
    # Get the directory name
    dir_name = os.path.dirname(gt_image) + os.path.sep
    
    # Make a dir with the GT image name
    image_masks_dir = os.path.join(dir_name + filename)
    os.makedirs(image_masks_dir, exist_ok=True)
    
    # Create subdirs to hold the gt_image and mask images
    # This structure is used by the NucleiDataset.load_mask
    image_dir = os.path.join(image_masks_dir, 'images')
    masks_dir = os.path.join(image_masks_dir, 'masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Copy the GT image into its respective folder
    shutil.copyfile(gt_image, os.path.join(image_dir, os.path.basename(gt_image)))
    
    print('------', os.path.join(image_dir, os.path.basename(gt_image)))
    
    with open(json_masks) as file:
        # Load json file
        json_annotations = json.load(file)
    
    # Convert json format to something more dict-like
    annotations = list(json_annotations.values())[0]
    
    # Extract polygon coord info
    polygons = [i for i in annotations['regions']]
    
    for i in range(0, len(polygons)):
        # Create an empty image to fill with mask info
        mask = np.zeros([height, width], dtype=np.uint8)
        # Enter dict key which contains polygon coords
        annotated_mask = polygons[i]['shape_attributes']
        # Extract x and y coords
        x_points = annotated_mask['all_points_x']
        y_points = annotated_mask['all_points_y']
        # Get vertices coords
        row_coords, col_coords = skimage.draw.polygon(y_points, x_points)
        # Turn polygon to 255 to make the mask white in the final image
        mask[row_coords, col_coords] = 255
        # Save image into masks folder
        skimage.io.imsave('{}/{}_mask_{}.png'.format(masks_dir, filename, i+1), mask)
        
    
# #%% Tests for json_to_png_mask()

# json_path = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/BBBC006-oof-u2os/images+masks/labelled_nuclei at z16.json"
# img_path = "/home/think/Documents/deep-click/datasets/nucleus/stats_images/BBBC006-oof-u2os/images+masks/oof-u2os.png"

# json_to_png_mask(img_path, json_path)

#%%

# json_path = "datasets/nucleus/label_test/1111_full-labelling.json"
# img_path = "datasets/nucleus/label_test/1111-full.png"

# json_to_png_mask(img_path, json_path)

def extract_masks(mask_image):
    """
    Identifies individual masks in a foreground/background image and extracts
    masks into individual images, for each mask, into a mask directory    
    """
    
    # Get the filename and remove the file extension
    filename = os.path.basename(mask_image).split(".")[0]
    
    # Get the parent directory name
    dir_name = os.path.dirname(mask_image) + os.path.sep
    
    # Make dir for images of masks
    mask_dir = os.path.join(dir_name, filename, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    
    # Open image
    img = skimage.io.imread(mask_image)

    # Flatten the array
    if np.ndim(img) == 3:
        flat_array = np.zeros(img.shape[0:2], dtype=np.int32)
        for i in range(0, img.shape[2]):
            # For each mask element, add to the 2d image array
            flat_array = flat_array + img[..., i]
    else:
        flat_array = img
    
    # Label gt masks
    labs = skimage.morphology.label(img)
    
    mask_min = np.min(labs[np.nonzero(labs)])
    mask_max = np.max(labs[np.nonzero(labs)])
    object_number = [obj for obj in range(mask_min, mask_max + 1)]  

    for i in object_number:
        skimage.io.imsave("{}/{}_mask_{}.png".format(mask_dir, filename, i), np.equal(labs, i))  



#%% Find and convert .tif files to .png

def get_filenames(directory):
    """
    For a given directory, find the filenames for files within subdirectories.
    """
    
    file_list = []
    # os.walk generator returns the path of the directories found, their names,
    # and all of the files found in the root and directories
    # 
    for (dir_path, dir_names, filenames) in os.walk(directory):
        file_list += [os.path.join(dir_path, file) for file in filenames]
        
    return file_list

def tif_to_png(directory):
    """
    Scans a given directory and creates a copy of any tif images in the png
    format
    """
    
    filenames = get_filenames(directory)
    
    tif_files = [img for img in filenames if ".tif" in img]
    
    for tif_img in tif_files:
        path = os.path.dirname(tif_img)
        filename = os.path.basename(tif_img).split(".")[0]
        save_path = os.path.join(path, filename+".png")
        
        _img = skimage.io.imread(tif_img)
        skimage.io.imsave(save_path, _img)
        

    