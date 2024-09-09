# code adapted from:
# https://www.analyticsvidhya.com/blog/2023/01/deploying-deep-learning-model-using-tkinter-and-pyinstaller/

import tkinter
from tkinter import filedialog, ttk
import tensorflow as tf
import os
from PIL import Image, ImageTk
import numpy as np
import torch

# labels used for predictions
labels = ['deepfake', 'faceswap', 'real']

# dictionary of model's names and the path where they are located
models = {
    'ResNet34': os.getcwd().__add__("/resnet34"),
    'ResNet50': os.getcwd().__add__("/resnet50"),
    'ResNet101': os.getcwd().__add__("/resnet101"),
    'ResNet152': os.getcwd().__add__("/resnet152"),
    'ResNet152_V2': os.getcwd().__add__("/resnet152_v2"),
    'Se_ResNet34': os.getcwd().__add__("/se_resnet34"),
    'Se_ResNet34_V2': os.getcwd().__add__("/se_resnet34_v2"),
    'Se_ResNet50': os.getcwd().__add__("/se_resnet50"),
    'Se_ResNet101': os.getcwd().__add__("/se_resnet101"),
    'Se_ResNet152': os.getcwd().__add__("/se_resnet152")
}

# create tkinter app, set size of 600x600 and title
app = tkinter.Tk()
app.geometry("600x600")
app.title("Deepfake Detection Model")

# create empty dictionary of loaded models
loaded_models = {}

# for each model in model dictionary, load the model using the path, and add pair into loaded_models dictionary
def load_all_models():
    for name, path in models.items():
        loaded_models[name] = tf.keras.models.load_model(path)
    print("All models loaded")

# current model is set to none
current_model = None

# change the current model based on user selection
def set_model(model_name):
    global current_model
    current_model = loaded_models[model_name]
    print(f"Model switched to: {model_name}")

# return path of image selected by the user
def getImage():
 path = filedialog.askopenfilename()
 return path

# calculate mean and std values that will be used to normalize the image.
# Image is converted to 'RGB' to follow the model setup
def getMeanandStd(img_path):
    image = Image.open(img_path)
    image = image.convert('RGB')
    image_array = np.array(image) / 255.0
    mean = np.mean(image_array, axis=(0, 1))
    std = np.std(image_array, axis=(0, 1))
    return mean, std

# apply transformations to image to reflect the transformations used in the training process.
def preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    mean, std = getMeanandStd(path)
    image = (image - mean) / std
    image = tf.expand_dims(image, axis=0)
    return image

# user selects image, which is then processed.
# The image is then turned into a numpy array and then into a tensor format.
# using the chosen model, predictions are made.
# the highest probability is chosen, and the mapped to the correct label.
# the image, predicted label and probability are displayed.
def image_prediction():
    path = getImage()
    if not path:  # Check if a path was selected
        return
    input_img = preprocess(path)
    input_array = input_img.numpy()
    input_tensor = torch.tensor(input_array)

    with torch.no_grad():
        outputs = current_model(input_tensor)

    outputs_array = outputs.numpy()
    predicted_label_index = np.argmax(outputs_array)
    predicted_label = labels[predicted_label_index]
    predicted_label_probability = np.max(outputs_array)

    img = Image.open(path)
    img = img.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    label_text.set(
        "Predicted label: {} (Probability: {:.2f})".format(predicted_label, predicted_label_probability))

# dropdown menu for model selection
model_selector = ttk.Combobox(app, values=list(models.keys()), state="readonly")
model_selector.pack(pady=20)
model_selector.bind('<<ComboboxSelected>>', lambda event: set_model(model_selector.get()))

# display image
image_label = tkinter.Label(app)
image_label.pack(pady=10)

# display prediction and probability
label_text = tkinter.StringVar()
prediction_label = tkinter.Label(app, textvariable=label_text)
prediction_label.pack(pady=10)

# upload image button
upload = tkinter.Button(app, text="Upload Image", command=image_prediction)
upload.pack(pady=10)

# all the models are loaded beforehand for less waiting time when running the application
load_all_models()

# start the application
app.mainloop()
