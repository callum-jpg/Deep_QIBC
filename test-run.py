import load_images
import detect
import process_images

#

import os
        
# Testing


# Load images
images = load_images.LoadImages()

images.add_channels(["w1DAPI", "w2GFP", "w3mCherry", "w4Cy5"])

image_dir = "/media/think/IA Ext HDD/Callum/27-3-20 pc local results/18-3-20 26-15 genotox ii/sorted/15-nt"

images.load_images(image_dir)

#%%

# Run detection
nuclei_detection = detect.DetectNucleus()

# Select channel to run detection on (in this case, DAPI)
object_channel = images.channels[0]

nuclei_detection.run_detection(images.image_info, object_channel, "low", "cpu")

nuclei_detection.results

#%%
# Load and label masks
labelled = process_images.ProcessMasks()

labelled.label_masks(nuclei_detection.results)

labelled.labelled_masks[0]['masks'].max()

#%%

intensity = process_images.RecordIntensity(images.image_info, images.channels, labelled.labelled_masks)

# Add itentified masks to image_info
intensity.consolidate_masks_with_images('w1DAPI')

#%%

data = intensity.record_intensity()

#%%

import pandas as pd

df = pd.DataFrame(data=data)

df.to_csv('test-run.csv', index=False)

#%% IT WORKKKKKKKKKKKSSSSSSSSSS

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 7))
ax.scatter(data['w1DAPI_total_intensity'], data["w4Cy5_mean_intensity"])
ax.set_xlim([10, 90])
ax.set_yscale('log')
ax.set_ylim([.0001, 0.04])

#%% Displaying a specific image
# Specifically, looking at objects identified for image 28
import skimage.color
import numpy as np


img = skimage.color.rgb2gray(intensity.image_info[27]['w1DAPI'][1])
img_masks = intensity.image_info[27]['masks']
rgb_masks = skimage.color.label2rgb(img_masks, bg_label = 0)

fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
ax[0].imshow(img, cmap="gray")
ax[1].imshow(rgb_masks)

#%%

fig, ax = plt.subplots(ncols=1, figsize=(15, 5))
ax.imshow(img, cmap="gray")

fig, ax = plt.subplots(ncols=1, figsize=(15, 5))
ax.imshow(rgb_masks)


