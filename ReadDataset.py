import pandas as pd
import numpy as np
import keras
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from glob import glob
from random import randint, seed, shuffle
import matplotlib.pyplot as plt

files = glob('/home/jerinpaul/Documents/Study Material/DISS1/Dataset/archive/*/*/*')
seed(10)
shuffle(files)
files = files[0:50000]
files = [f for f in files if f.endswith('png')]
labels = [int(f[-5]) for f in files]
print(labels.count(1)/len(files))

#function to load image
def load_data(files, index):
    X = []
    img = load_img(files[index], target_size = (50,50))
    pixels = img_to_array(img)
    pixels = pixels/255
    X.append(pixels)
    return np.stack(X)

#function to show image
def show_image(img, cm):
    plt.imshow(img, cmap = cm)
    if cm!=None:
        print(f'color map: {cm}')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def featureExtraction():
    model = ResNet50(weights="imagenet", include_top=False)
    batch_size = 32
    col_names = ['serial_num', 'truth_val']
    for i in range(8192):
        col_names.append(i + 1)

    image_feature_df = pd.DataFrame(columns = col_names)
    for (b, i) in enumerate(range(0, len(files), batch_size)):
        print("[INFO] processing batch {}/{}".format(b + 1, int(np.ceil(len(files) / float(batch_size)))))

        batchPaths = files[i:i + batch_size]
        batchLabels = labels[i:i + batch_size]
        batchImages = []

        for imagePath in batchPaths:
            image = load_img(imagePath, target_size=(50, 50))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=batch_size)
        features = features.reshape((features.shape[0], 2 * 2 * 2048))

        for (c, index) in enumerate(features):
            feature_ls = [c + i + 1, batchLabels[c]]
            for n in index:
                feature_ls.append(n)
            image_feature_df.loc[len(image_feature_df)] = feature_ls

    image_feature_df.to_csv("/home/jerinpaul/Documents/Study Material/DISS1/Dataset/image_feature.csv")

#X = load_data(files, 0)
featureExtraction()
csvFile = pd.read_csv('/home/jerinpaul/Documents/Study Material/DISS1/Dataset/image_feature.csv')
print(csvFile)
