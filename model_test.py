from cProfile import label
import sklearn
import numpy as np
import pandas as pd
from keras_preprocessing import image
import tensorflow as tf
import os
import csv
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('weights-improvement-22-0.93')

images = image.load_img("gau-meo.jpeg", target_size=(150, 150))
images = image.img_to_array(images)
images = np.expand_dims(images, axis=0)



a = model.predict(images, batch_size=32)
print(a)
if a[0][0] > a[0][1]:
    print("gato")
    a = "cat"
else:
    print("perro")
    a = "dog"










