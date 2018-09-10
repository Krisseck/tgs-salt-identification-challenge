import numpy as np
import random
import math
import os.path
from keras.models import Sequential, load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

source_csv = 'depths.csv'
source_csv_delimiter = ','

# fix random seed for reproducibility
np.random.seed(7)

model = load_model('test.h5')

# Encode predicted masks for submission
def RLenc(pixels):
  encoding = []
  start = None
  left = right = 0
  for i in range(len(pixels)):
    left = right
    right = pixels[i]
    
    if (left==0) & (right==1):
      start = i+1
      encoding.append(start)
    if (left==1) & (right==0):
      encoding.append(i-start+1)
    
  # Final check
  if right==1:
    encoding.append(i-start+2)

  def to_str(a):
    s = str(a[0])
    for i in range(1,len(a)):
      s = s + ' ' + str(a[i])
    return s
    
  if len(encoding)==0:
    return ''
  else:
    return to_str(encoding) 

depths = np.genfromtxt(source_csv, delimiter=source_csv_delimiter, skip_header=1, dtype=['U16', 'float_'])

for i in range(len(depths)):
  depths[i][1] /= 1000

print('id,rle_mask')

inputs = []
inputIds = []

for i in range(len(depths)):

  if(os.path.isfile('test/images/' + depths[i][0] + '.png')):

    image = load_img('test/images/' + depths[i][0] + '.png', color_mode='grayscale')
    image_array = img_to_array(image)
    image_array /= 255

    image_array = np.insert(image_array, 0, depths[i][1], 2)

    image_array = np.expand_dims(image_array, axis=0)

    inputs.append(image_array)
    inputIds.append(depths[i][0])


inputs = np.array(inputs)

inputs = np.squeeze(inputs, axis=1)

predictions = model.predict(inputs)

for i in range(len(inputs)):

  prediction = predictions[i]

  prediction[prediction >= 0.5] = 1
  prediction[prediction < 0.5] = 0

  print(inputIds[i] + ',' + RLenc(prediction))