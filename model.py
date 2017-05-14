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
from sklearn.model_selection import train_test_split
from keras import backend as K

OFFCENTER_CAMERA_STEERING_CORRECTION = 0.2
BATCH_SIZE = 32

def gen_imgs(samples, batch_size):
    '''Train '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = samples.sample(frac=1) # shuffle samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]

            images = []
            angles = []
            for _, batch_sample in batch_samples.iterrows():
                img = cv2.cvtColor(cv2.imread(str(batch_sample.img_path)), cv2.COLOR_BGR2RGB)
                angle = batch_sample.steering

                if batch_sample.camera_position.startswith('left'):
                    steering_correction = OFFCENTER_CAMERA_STEERING_CORRECTION
                elif batch_sample.camera_position.startswith('right'):
                    steering_correction = -OFFCENTER_CAMERA_STEERING_CORRECTION
                else:
                    steering_correction = 0.0

                if batch_sample.flip_img:
                    img = cv2.flip(img, 1)
                    angle = angle * -1.0
                    steering_correction  * -1.0

                angle += steering_correction

                images.append(img)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

def load_samples(input_dir):
    headers = ('center_image', 'left_image', 'right_image',
        'steering', 'throttle', 'brake', 'speed')
    samples = pd.read_csv(input_dir / 'driving_log.csv', header=None, names=headers)

    # Training images were generated on Windows machine, fix paths to work on Linux/MacOS machines
    fix_path = lambda p: input_dir / 'IMG' / PureWindowsPath(p).parts[-1]

    for col in ['center_image', 'left_image', 'right_image']:
        samples[col]= samples[col].apply(fix_path)
    return samples

base_input_dir = Path('data')
data_dirs = ('track1', 'track1_reversed', 'track2', 'track2_reversed')
samples = pd.concat([load_samples(base_input_dir / p) for p in data_dirs],
    ignore_index=True)

# Drop unneccessary columns and reshape dataframe
samples.drop(['throttle', 'brake', 'speed'], axis=1, inplace=True)
samples = pd.melt(samples, id_vars=['steering'],
    value_vars=['center_image', 'left_image', 'right_image'],
    var_name='camera_position', value_name='img_path')
samples['camera_position'] = samples.camera_position.str.replace('_image', '')

# Augment data with flipped images
samples['flip_img'] = False # flag to let gen_imgs know whether to flip image 
samples_flipped = samples.copy()
samples['flip_img'] = True
samples = pd.concat([samples, samples_flipped], ignore_index=True)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = gen_imgs(train_samples, batch_size=BATCH_SIZE)
validation_generator = gen_imgs(validation_samples, batch_size=BATCH_SIZE)



# Define network architecture
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

model.fit_generator(train_generator, np.ceil(len(train_samples) / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE),
    epochs=3)

model.save('model.h5')
