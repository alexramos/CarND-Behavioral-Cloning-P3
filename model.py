from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# load data

input_dir = Path('data/IMG/')
headers = ['center_image', 'left_image', 'right_image',
	'steering', 'throttle', 'brake', 'speed']
df = pd.read_csv('data/driving_log.csv', header=None, names=headers) #, skiprows=1)
df.brake = 0

fix_path = lambda p: input_dir / PureWindowsPath(p).parts[-1]
for col in ['center_image', 'left_image', 'right_image']:
	df[col]= df[col].apply(fix_path)
    # df[col] = 'data/' + df[col]

OFFCENTER_CORRECTION = 0.2

load_img = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)

images = list()
measurements = list()
print("Loading center images...")
for i, img_path in enumerate(tqdm(df.center_image)):
    img = load_img(img_path)
    images.append(img)
    measurements.append(df.steering.iloc[i])
    # augment data with flipped images
    images.append(cv2.flip(img, 1)) 
    measurements.append(df.steering.iloc[i] * -1.0)

print("Loading left images...")
for i, img_path in enumerate(tqdm(df.left_image)):
    img = load_img(img_path)
    images.append(img)
    measurements.append(df.steering.iloc[i] + OFFCENTER_CORRECTION)
    # augment data with flipped images
    images.append(cv2.flip(img, 1)) 
    measurements.append(df.steering.iloc[i] * -1.0 - OFFCENTER_CORRECTION)

print("Loading right images...")
for i, img_path in enumerate(tqdm(df.right_image)):
    img = load_img(img_path)
    images.append(img)
    measurements.append(df.steering.iloc[i] - OFFCENTER_CORRECTION)
    # augment data with flipped images
    images.append(cv2.flip(img, 1)) 
    measurements.append(df.steering.iloc[i] * -1.0 + OFFCENTER_CORRECTION)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,
    shuffle=True, epochs=3)

model.save('model.h5')

# preprocess
# img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.show()

# - crop
# - resize?
# - change colors

# create model
# - use nvida architecture
# - model with keras

# train model
 
# save model to h5 file
# check out https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
