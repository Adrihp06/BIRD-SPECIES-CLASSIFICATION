#import the libraries to create the CNN
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import EfficientNetB0
import time
import tensorflow.keras

# Use cv2 to read the bird folder and image, each folder is the name of the bird
DATADIR = "/app/CNN/"
train_dir = "/app/CNN/train"
test_dir = "/app/CNN/test"
validation_dir = "/app/CNN/valid"
CATEGORIES = [i for i in os.listdir(train_dir)]
IMG_DIM = (220,220)
#Data augmentation
#use flow from direcory to read the images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_DIM, batch_size=32,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_DIM, batch_size=32,
    class_mode="categorical"
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=IMG_DIM, batch_size=32,
    class_mode="categorical"
)

#load the model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(220,220,3))
#freeze the layers

#Unfreeze the last 20 layers
base_model.trainable = True
'''set_trainable = False
for layer in base_model.layers:
    if layer.name.startswith('top_'):
        set_trainable = True
    elif layer.name.startswith('block7a'):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False'''
#Unfreeze the block 7

#add the layer
output = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
output = Dense(450, activation='softmax')(output)
#create the model
model = Model(inputs=base_model.input, outputs=output)
model.summary()

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
history = model.fit(train_generator, steps_per_epoch = 1000, epochs = 10, validation_data = validation_generator, validation_steps = 50)

#metrics
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#test the model
test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)

