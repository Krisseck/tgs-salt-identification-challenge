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

trainX = np.load('dataset_train_x.npy')
trainY = np.load('dataset_train_y.npy')

# make it divisable by batch size
remainder = len(trainX) % batch_size
if remainder > 0:
  trainX = trainX[:-remainder]
  trainY = trainY[:-remainder]

print(trainX.shape)
print(trainY.shape)

# create and fit model
model = Sequential()

model.add(Dense(2, input_shape=(101, 101, 2), activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10201, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

earlystopper = EarlyStopping(monitor='val_loss', verbose=1, patience=5)
checkpointer = ModelCheckpoint(filepath='test.h5', verbose=1, save_best_only=True)
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[checkpointer, earlystopper])