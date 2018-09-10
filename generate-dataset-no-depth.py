import numpy as np
import random
import math
import os.path
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, Flatten
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import array_to_img, img_to_array, load_img

epochs = 1000
batch_size = 1
validation_split = 0.2

source_csv = 'depths.csv'
source_csv_delimiter = ','

# fix random seed for reproducibility
np.random.seed(7)

depths = np.genfromtxt(source_csv, delimiter=source_csv_delimiter, skip_header=1, dtype=['U16', 'float_'])

for i in range(len(depths)):
  depths[i][1] /= 1000

trainX = []
trainY = []

for i in range(len(depths)):

  if(os.path.isfile('train/images/' + depths[i][0] + '.png')):

    image = load_img('train/images/' + depths[i][0] + '.png', color_mode='grayscale')
    image_array = img_to_array(image)
    image_array /= 255

    mask = load_img('train/masks/' + depths[i][0] + '.png', color_mode='grayscale')
    mask_array = img_to_array(mask)
    mask_array /= 255
    mask_array = np.array(mask_array)
    mask_array = mask_array.flatten()

    trainX.append(image_array)

    trainY.append(mask_array)

trainX = np.array(trainX)
trainY = np.array(trainY)

print(trainX[40])
print(trainY[40])

# make it divisable by batch size
remainder = len(trainX) % batch_size
if remainder > 0:
  trainX = trainX[:-remainder]
  trainY = trainY[:-remainder]

print(trainX.shape)
print(trainY.shape)

np.save('dataset_train_x_no_depth', trainX)
np.save('dataset_train_y_no_depth', trainY) 

'''

# create and fit model
model = Sequential()

model.add(Dense(5, input_shape=(2, 101, 101, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10201, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

earlystopper = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
checkpointer = ModelCheckpoint(filepath='test.h5', verbose=1, save_best_only=True)
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[checkpointer, earlystopper])
'''