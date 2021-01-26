import os

from difflib import get_close_matches




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
    
    
    def match_images(self, filelist):
        """
        Performs a fuzzy string match on a filename list.
        Strings are matched into lists with the same length as self.channels.
        Removes channel strings from filename to aid in matching.
        """
        
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
        
    def sort_images(self, image_set):
        
        output_dict = {}
        
        for channel_img in image_set:
            for channel in self.channels:
                if channel in channel_img:
                    output_dict.update({channel: channel_img})
        return output_dict
        
    
    def add_images(self):
        """
        Add grouped images to dict
        """
        
        
        for image_id, image_set in enumerate(self.grouped_images):
            image_info = {}
            #print(image_set)
            image_info.update({"image number": image_id+1})
            image_info.update(self.sort_images(image_set)) 
            self.image_info.append(image_info)
            # for channel_img in image_set:
                # for channel in self.channels:
                #     if channel in channel_img:
                #         #print(channel_img)
                #         image_info.update({channel: channel_img})
                

                    

    
        
        
        
#%% Testing

images = LoadImages()

images.add_channels(["w1DAPI", "w2Cy5", "w3mCherry", "w4GFP"])

images.match_images(os.listdir("../../datasets/nucleus/images"))

images.add_images()

print(images.image_info)




#%%

test = {"key1": 1, "key2": 2}

test.update({"key3": 3, "key4": 4})


#%%

fl = os.listdir("../../datasets/nucleus/images")
fl.sort()


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