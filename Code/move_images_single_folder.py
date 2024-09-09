# the script to extract frames from images creates a new folder for each frame
# this script takes every image and moves it inside one single folder

import os
import shutil

# change directory to relevant folders
source_folder_fake = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Dataset/manipulated_sequences/Deepfakes/c23/videos/images"
destination_folder_fake = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Dataset/manipulated_sequences/Deepfakes/c23/videos/images_sorted"

source_folder_true = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Dataset/original_sequences/youtube/c23/videos/images"
destination_folder_true = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Dataset/original_sequences/youtube/c23/videos/images_sorted"

for root, dirs, files in os.walk(source_folder_true):
    for file_name in files:
        # construct full file path
        source = os.path.join(root, file_name)
        # define the destination file path
        destination = os.path.join(destination_folder_true, file_name)
        
        # copy file
        shutil.copy(source, destination)
        print('Copied', file_name)
        
for root, dirs, files in os.walk(source_folder_fake):
    for file_name in files:
        # construct full file path
        source = os.path.join(root, file_name)
        # define the destination file path
        destination = os.path.join(destination_folder_fake, file_name)
        
        # copy file
        shutil.copy(source, destination)
        print('Copied', file_name)