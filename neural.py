import sklearn
import numpy as np
import pandas as pd
from keras_preprocessing import image
import tensorflow as tf
import os
import csv
from sklearn.neural_network import MLPClassifier
import pickle

a = pd.read_csv("n.csv")
a = pd.DataFrame(a)
y = np.array(a)

x_img = []

for nombre in os.listdir(r"dogs-vs-cats\train\train"):
    images = image.load_img(r"dogs-vs-cats\train\train\\" + nombre, target_size=(150, 150))
    
    x = image.img_to_array(images)
    x = tf.image.rgb_to_grayscale(x)
    x = np.expand_dims(x, axis=0)
    x_img.append(x)

x = np.array(x_img).reshape(150*150, -1)

model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
model.fit(x, y)

# Save the model to disk
filename = 'm_test.pkl'
pickle.dump(model, open(filename, 'wb'))



