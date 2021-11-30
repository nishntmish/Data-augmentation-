# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.preprocessing.image import ImageDataGenerator,load_img,array_to_img,img_to_array
import numpy 

datagen= ImageDataGenerator(rotation_range=20,
                            height_shift_range=0.2,
                            width_shift_range=0.2,
                            zoom_range=0.2,
                            shear_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

img = load_img('D:\DATA SCIENCE DATA SETS\CNN\Data Augmentaion\Virat-Kohli-RCB.jpg')


x= img_to_array(img)

x= x.reshape((1,) + x.shape)


i=0
for batch in datagen.flow(x,
                          batch_size=1,
                          save_format='jpeg',
                          save_prefix='virat',
                          save_to_dir='preview'):
    i+=1
    if i > 20:
        break

          


