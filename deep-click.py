import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from skimage import io # for io.imread
from skimage.filters import threshold_otsu, rank, threshold_multiotsu
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border

from scipy import ndimage

#%% Load image

filename = "images/c15-gfp-yh2ax-pcna-(nt-hu-cpt-mms-dt_neg-dt_pos-ea_neg-ea_pos)_w1DAPI_s269.TIF"

# imread returns a np array for containing values for each pixel
# np.shape(img) = (height, width) in px
img = io.imread(filename)

# display
plt.imshow(img, cmap='gray')


#%% Check histogram
# http://morphogenie.fr/segmenting-nuclei.html

bit_range = (0, 4096)

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(img.flatten(), log=True, bins = bit_range[1], range = bit_range)

ax.set_title("min: {0} \n"
             "max: {1}"
             .format(img.min(), img.max())
             )


#%% Threshold

smooth_size = 5
min_rad = 10
max_rad = 100

# Calculate Otsu threshold for single image
#threshold = threshold_otsu(img.max(axis=0))
threshold = 500


binary = img > threshold

dist = ndimage.distance_transform_edt(binary)

local_maxi = peak_local_max(dist, indices=False, labels=binary, min_distance=2*min_rad)

markers = ndimage.label(local_maxi)[0]

labelled_img = watershed(-dist, markers, mask=binary)

#%% Plot objects

fig, ax = plt.subplots(1, 3)

ax[0].imshow(img, cmap='gray')
ax[1].imshow(-dist, cmap='gray')
ax[2].imshow(labelled_img, cmap='gray')


#%% Measure stuff in labelled region

# regionprops doesn't allow for sum of pixel intensities

output = []
cols = ['nuclei_number', 'area', 'mean_intensity']

values = regionprops(labelled_img.astype(np.int), intensity_image=img)

for obj in values:
    output.append([obj.label,
                   obj.area,
                   obj.mean_intensity
                   ])

df = pd.DataFrame(output, columns=cols)

df.head()



label_image = label(labelled_img)

image_label_overlay = label2rgb(label_image, image=img, bg_label=0)

fig, ax = plt.subplots()

ax.imshow(image_label_overlay)


#%% https://exeter-data-analytics.github.io/python-data/skimage.html
#%% Alternative measuring of stuff. regionprops independent and allows for use of ndimage


# Incrementally label nuclei based on watershed 
obj_labels = label(labelled_img) 




nuclei_number = [obj
                 for obj in range(1, obj_labels.max()+1)]

mean_intensity = [ndimage.mean(img, obj_labels == obj)
                  for obj in range(1, obj_labels.max()+1)]

total_intensity = [ndimage.sum(img, obj_labels == obj)
                   for obj in range(1, obj_labels.max()+1)]

area = [(obj_labels == obj).sum() 
          for obj in range(1, obj_labels.max()+1)] 

# sum_div_area is a test to ensure that mean is the total intensity in a object
# divided by the total number of pixels
sum_div_area = [ndimage.sum(img, obj_labels == obj) / (obj_labels == obj).sum()
                for obj in range(1, obj_labels.max() + 1)]

# Create a list of lists for all data
data = [] 
data.append(nuclei_number)
data.append(mean_intensity)
data.append(total_intensity)
data.append(area)
data.append(sum_div_area)

# Declare column names, in order of list of list elements
cols = ['nuc_number', 'mean', 'total', 'area', 'sum_div_area']

# Transpose writes columns as rows 
# Since list of lists creates a len(list) wide df
df = pd.DataFrame.from_dict(data).transpose()
df.columns = cols
df[cols[0]] = df[cols[0]].astype(np.int64) # Remove trailing .0 added by pandas


#%% 
###### Improving nuclei object detection

filename = "images/c15-gfp-yh2ax-pcna-(nt-hu-cpt-mms-dt_neg-dt_pos-ea_neg-ea_pos)_w1DAPI_s269.TIF"

# imread returns a np array for containing values for each pixel
# np.shape(img) = (height, width) in px
img = io.imread(filename)

# display
# plt.imshow(img, cmap='gray')

# threshold = threshold_multiotsu(img, classes=3)
# # For a given pixel intensity (ie. value of array), determine which threshold 
# # bin the value belongs to. Eg. for bins ([100, 400]) a pixel intensity of 
# # 20 would belong to bin 0 (since it's less than 100), 200 would belong to bin
# # 1, and 700 to bin 3. Bin values are the index of the threshold. np.digitize
# # creates an array of equal shape and and catetegorises each pixel into these 
# # bins.
# regions = np.digitize(img, bins=threshold)

threshold = threshold_otsu(img)
regions = img > threshold

# fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
# ax[0].imshow(img)
# ax[0].set_title('Original')
# ax[0].axis('off')
# ax[1].imshow(regions)
# ax[1].set_title('Otsu thresholding')
# ax[1].axis('off')


# Mean calculates the local means of an image. This removes intensity 'holes'
# img_as_ubyte converts bool array to 8-bit (False becomes 0, True becomes 255)
# disk creates a disk shaped struturing element with radius 4
smooth = rank.mean(img_as_ubyte(regions), disk(4))

# Convert disk regions to 
binary_smooth = smooth > 20

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(regions)
ax[1].imshow(binary_smooth)

# Number of objects
label(binary_smooth).max()

#%%

# Watershed
distance = ndimage.distance_transform_edt(binary_smooth)

local_maxi = peak_local_max(distance, indices=False,
                                        min_distance=15)
markers = label(local_maxi)
segmented_cells = watershed(-distance, markers, mask=binary_smooth)

colour_labs = label2rgb(segmented_cells, bg_label=0)

fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
ax[0].imshow(img, cmap='gray')
ax[1].imshow(colour_labs)

# number of objects post watershed
label(colour_labs).max()



#%% 
############# All together now

filename = "images/c15-gfp-yh2ax-pcna-(nt-hu-cpt-mms-dt_neg-dt_pos-ea_neg-ea_pos)_w1DAPI_s269.TIF"

# imread returns a np array for containing values for each pixel
# np.shape(img) = (height, width) in px
img = io.imread(filename)


# Calculate Otsu threshold to select background and foreground
threshold = threshold_otsu(img)
# Find pixel intensities (array value) which are greater than the threshold. 
# Returns True/False
obj = img > threshold

# Remove border touching objects
obj = clear_border(obj)


# This removes intensity 'holes'
# Mean calculates the local means of an image
# img_as_ubyte converts bool array to 8-bit (False becomes 0, True becomes 255)
# disk creates a disk shaped struturing element with radius 4
smooth = rank.mean(img_as_ubyte(obj), disk(4))

# Convert disk regions to True/False. Values are either 0 or 255 since 8-bit
binary_smooth = smooth > 20

# Visualise original objects vs smoothed
# fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(obj)
# ax[2].imshow(binary_smooth)

# Watershed
# Calculate watershed to split overlapping obj
distance = ndimage.distance_transform_edt(binary_smooth)

# Min distance can change based on magnification
local_maxi = peak_local_max(distance, indices=False,
                                        min_distance=15)


markers = label(local_maxi)
segmented_obj = watershed(-distance, markers, mask=binary_smooth)

colour_labs = label2rgb(segmented_obj, bg_label=0)

# Plot original image and watershed split images
# fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(binary_smooth)
# ax[2].imshow(colour_labs)

#%%
# Measuring pixel intensities

# Incrementally label nuclei based on watershed 
obj_labels = label(segmented_obj) 


nuclei_number = [obj
                 for obj in range(1, obj_labels.max()+1)]

mean_intensity = [ndimage.mean(img, obj_labels == obj)
                  for obj in range(1, obj_labels.max()+1)]

total_intensity = [ndimage.sum(img, obj_labels == obj)
                   for obj in range(1, obj_labels.max()+1)]

area = [(obj_labels == obj).sum() 
          for obj in range(1, obj_labels.max()+1)] 

# sum_div_area is a test to ensure that mean is the total intensity in a object
# divided by the total number of pixels
sum_div_area = [ndimage.sum(img, obj_labels == obj) / (obj_labels == obj).sum()
                for obj in range(1, obj_labels.max() + 1)]

# Create a list of lists for all data
data = [] 
data.append(nuclei_number)
data.append(mean_intensity)
data.append(total_intensity)
data.append(area)
data.append(sum_div_area)

# Declare column names, in order of list of list elements
cols = ['nuc_number', 'mean', 'total', 'area', 'sum_div_area']

# Transpose writes columns as rows 
# Since list of lists creates a len(list) wide df
df = pd.DataFrame.from_dict(data).transpose()
df.columns = cols
df[cols[0]] = df[cols[0]].astype(np.int64) # Remove trailing .0 added by pandas if ya want

#%% Filter objects given a min and max diameter

min_size = 5
max_size = 100

min_area = np.pi * (min_size**2) / 4
max_area = np.pi * (max_size**2) / 4

# from CP
# area_image = areas[labeled_image]
# labeled_image[area_image < min_allowed_area] = 0
# small_removed_labels = labeled_image.copy()
# labeled_image[area_image > max_allowed_area] = 0


#%% Loading multiple images

from natsort import natsorted

image_dir = 'images'

input_images = natsorted(os.listdir(image_dir))

list1 = []

for image in input_images:
    path = os.path.join(image_dir, image)
    list1.append(io.imread(path))


#%% 3d array

channels = 4
ch1 = 'DAPI'
ch2 = 'GFP'
ch3 = 'mCherry'
ch4 = 'Cy5'

ch_list = [ch1, ch2, ch3, ch4]

for image in input_images:
    for channel in ch_list:
        if channel in image:
            print(image, 'DDDD', channel)


# Create an array with 4 'channels', each with a 3x3 image
y = np.zeros((4, 3, 3))

for i in range(len(y)):
    y[i, 0, 0] = i + 1


# In a 3d array, axis 0 = number of channels, and axis 1 and 2 represent 
# the height and width, respectively

# For a given number of channels, create am empty 3d array (axis0=channels) 
# for each image
z = np.zeros((channels, io.imread(path).shape[0], io.imread(path).shape[1]))

# Add specific image pixel data to image
z[0] = io.imread(path)


image_dir = 'images'

input_images = natsorted(os.listdir(image_dir))

img = []

img_arr = np.zeros((channels, 
                    io.imread(path).shape[0],
                    io.imread(path).shape[1]))

for image in input_images:
    for i, channel in enumerate(ch_list):
        if channel in image:
            path = os.path.join(image_dir, image)
            img_arr[i] = io.imread(path)
img.append(img_arr)


#%%

from difflib import get_close_matches

ch1 = 'w1DAPI'
ch2 = 'w2GFP'
ch3 = 'w3mCherry'
ch4 = 'w4Cy5'

ch_list = [ch1, ch2, ch3, ch4]

# Remove channel info from filename
test = [img.replace(ch, '') for ch in ch_list for img in input_images if ch in img]

f = get_close_matches(test[0], input_images, n=4, cutoff=0.6)

#%%

from difflib import get_close_matches

ch1 = 'w1DAPI'
ch2 = 'w2Cy5'
ch3 = 'w3mCherry'
ch4 = 'w4GFP'

ch_list = [ch1, ch2, ch3, ch4]

image_dir = 'images'

input_image_list = os.listdir(image_dir)

input_image_path = [os.path.join(image_dir, i) for i in input_image_list]

def file_group(filelist, non_group, group_size):
    """
    Performs a fuzzy string match on a list.
    Strings are matched into lists of length group_size.
    Removes substrings, non_group, which are to be ignored for grouping.
    
    Parameters
    ----------   
    filelist : list
        List of strings to be grouped
    
    non_group : list
        List of strings found in filelist to not be used for matching
        
    group_size : int
        Returning group size
    
    Returns
    -------
        out : nested list
            A list of lists, with each sublist representing a group of matched strings
    """
    
    grouped_list = []
    
    # Remove non_group from elements of list
    strip_list = [file.replace(ng, '') for ng in non_group for file in filelist if ng in file]

    # Identify unique elements
    unique = list(set(strip_list))

    # Iterate through unique elements and find fuzzy matches in filelist
    for i in unique:
        sub_list = get_close_matches(i, filelist, n=group_size, cutoff=0.9)
        grouped_list.append(sub_list)   
        
    # Make sure it returns an equal number of elements to the original filelist
    if len(filelist) != sum(len(x) for x in grouped_list):
        #return "Images inaccurately grouped"
        raise ValueError("Images unable to be accurately grouped")
    else:
        return grouped_list
    

grouped_filelist = file_group(input_image_path, ch_list, 4)

#%%

def read_image_channels(filelist_path, channels):
    """
    
    """



    # Empty list to hold 3d arrays of images
    img = []
    
    for image in filelist_path:
        # Create an empty 3d array with a shape of channels, image height, and image width
        # Based on first image
        img_arr = np.zeros((len(channels),
                            io.imread(filelist_path[0][0]).shape[0],
                            io.imread(filelist_path[0][0]).shape[1]))
        #print(image)
        for channel_img in image:
            #print(channel_img)
            for i, channel_name in enumerate(channels):
                if channel_name in channel_img:
                    img_arr[i] = io.imread(channel_img)
        img.append(img_arr)

    return img


input_images = read_image_channels(grouped_filelist, ch_list)

# plt.imshow(q[0][1], cmap='gray')

# Number of images
len(q)

# (number of channels, height, width)
q[0].shape


#%% Identify nuclei

# Create a function to get position/index of a particular channel
# Will allow for the returned index to be able to slice the img np.array

test = [q[0][0], q[1][0]]
np.concatenate(test)
threshold_otsu(np.concatenate(test))

def calculate_otsu_threshold(image_array, nuclei_channel):
    """
    """
    # Select the index for the given channel from ch_list
    nuclei_index = [(i, j) for i, j in enumerate(ch_list) if nuclei_channel in j][0][0]
    
    threshold_list = []
    
    for image in image_array:
        threshold_list.append(image[nuclei_index])
    
    #print(threshold_list)
    
    return threshold_otsu(np.concatenate(threshold_list, axis=0))

image_threshold = calculate_otsu_threshold(input_images, ch_list[0])


#%%

def identify_nuclei(image_array, nuclei_channel):
    """

    """
    nuclei_index = [(i, j) for i, j in enumerate(ch_list) if nuclei_channel in j][0][0]
    
    threshold = calculate_otsu_threshold(image_array, nuclei_channel)
    
    for img in image_array:
        # Find pixels above threshold
        thresh_img = img[nuclei_index] > threshold
        
        # Clear object border
        
        # Smooth intensity, then binary_smooth
        
        # calculate watershed
        
        # then label watershed. These are the identified nuclei
        
        # Store nuclei masks in list of tuples
        # (image_number, np.array)
        # eg [(0, np.arr), (1, np.arr)...]
                    
                
identify_nuclei(input_images, ch_list[0])

# Next, use labelled nuclei array to extract intensity info from full channel img array

#%% 

# threshold
# watershed nuclei
# label nuclei


for i in grouped_filelist:
    #print(i)
    for j in i:
        print(j)



#%%
test = io.imread(grouped_filelist[0][0])



