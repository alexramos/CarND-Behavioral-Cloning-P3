from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# load data

input_dir = Path('data/IMG/')
headers = ['center_image', 'left_image', 'right_image',
	'steering', 'throttle', 'brake', 'speed']
df = pd.read_csv('data/driving_log.csv', header=None, names=headers)
df.brake = 0

fix_path = lambda p: input_dir / PureWindowsPath(p).parts[-1]
load_img = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
for col in ['center_image', 'left_image', 'right_image']:
	df[col]= df[col].apply(fix_path)

images = list()
for img_path in tqdm(df.center_image):
	images.append(load_img(img_path))


X_train = np.array(images)
y_train = df.steering.values



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
