#import the libraries to create the CNN
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import time
import tensorflow.keras

# Use cv2 to read the bird folder and image, each folder is the name of the bird
DATADIR = "/app/"
train_dir = "/app/train"
test_dir = "/app/test"
validation_dir = "/app/valid"
CATEGORIES = [i for i in os.listdir(train_dir)]
IMG_DIM = (220,220)
#Data augmentation
#use flow from direcory to read the images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_DIM, batch_size=128,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_DIM, batch_size=128,
    class_mode="categorical"
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=IMG_DIM, batch_size=128,
    class_mode="categorical"
)

#create the model
input = Input(shape=(220,220,3))
x = Conv2D(32, (3,3), activation='relu')(input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = Conv2D(256, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
output = Dense(450, activation='softmax')(x)

model = Model(input, output)
model.summary()

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model
history = model.fit(train_generator, steps_per_epoch = train_generator.samples*2/128, epochs = 20, validation_data = validation_generator, validation_steps = 50, workers = 2)

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

