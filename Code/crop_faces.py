# this function is used to crop faces from images in a given dataset and create a new dataset maintaing the labels order

import os
import cv2
from skimage import io, color, img_as_ubyte
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

# change path to where the current images are
dataset_path = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Personal_Dataset"
# change path to where the new dataset will be created
new_dataset_path = "/Users/apple/Desktop/uni/year3/term2/PersonalProject/Cropped_Personal_Dataset"

# OpenCV's pre-trained Haar cascade model. Used to detect faces in images/videos.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# loop through each file in the directory
for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        # only deal with .jpg files
        if filename.endswith(".jpg"):
            
            # set folder name
            folder = os.path.basename(root)
            
            # read file, turn to grayscale and convert to unsigned byte format
            img = io.imread(os.path.join(root, filename))
            img_gray = color.rgb2gray(img)
            img_gray = img_as_ubyte(img_gray)
            
            # use pre-trained model to detect faces
            faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
            
            # use a counter in case there are multiple faces detect in a single image
            counter = 1
            
            for face in faces:
                
                # create new file name
                counter+=1
                new_filename = str(counter)+filename
                
                # get coordinates of face, and crop image and convert to color.
                x, y, w, h = face[0], face[1], face[2], face[3]
                face_box = (x, y, x+w, y+h)
                cropped_img = img[y:y+h, x:x+w]
                face_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                
                # save image in new dataset
                cv2.imwrite(os.path.join(new_dataset_path, folder, new_filename), face_img)
                print(folder, new_filename, "copied")
            
            
           


