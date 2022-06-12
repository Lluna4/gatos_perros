import sklearn
import numpy as np
import pandas as pd
from keras_preprocessing import image
import tensorflow as tf
import os
import csv
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras.callbacks import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
ddir = r"kagglecatsanddogs_5340\PetImages\\"
training_datagen = image.ImageDataGenerator(rescale=1./255)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])




train_generator = training_datagen.flow_from_directory(
    ddir,
    target_size=(150, 150),
    class_mode="categorical",
    shuffle=True,
)





model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape = (150, 150, 3) , data_format= "channels_last"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax'),

])
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

filepath="test/weights-improvement-{epoch:02d}-{accuracy:.2f}"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(train_generator, epochs=25, verbose=1, callbacks=callbacks_list, batch_size=32)
model.save("modelo3")



