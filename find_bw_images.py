import sys
import os
import pandas as pd

#%%

classes = pd.read_csv("classes.csv")

# bw = [filename for (fore, back) in ]

row = next(classes.iterrows())[1]

row['foreground']


bw = []
for i, j in classes.iterrows():
    if (j['foreground'] == "white") & (j["background"] == "black"):
        # Extract the BW image filename (same name as parent direcotry)
        bw.append(os.path.splitext(j["filename"])[0])
        
#%% Move BW images to new folder

import shutil

from distutils.dir_util import copy_tree

img_dir = next(os.walk('stage1_train'))[1]

dest = os.path.join(os.getcwd(), "bw_train")

for img in img_dir:
    if img in bw:
        # Get BW images src path name
        src = os.path.join(os.getcwd(), 'stage1_train', img)
        # Save in direcotry with image name
        dst = os.path.join(os.getcwd(), dest, img)
        copy_tree(src, dst)
        print(src)
