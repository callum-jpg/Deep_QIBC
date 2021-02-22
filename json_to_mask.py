import os
import skimage.io
import skimage.draw
import json
import numpy as np 
import shutil
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
        
    
#%%

json_path = "datasets/nucleus/label_test/1111_annotations.json"
img_path = "datasets/nucleus/label_test/1111.png"

json_to_png_mask(img_path, json_path)