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

a = pd.read_csv("n.csv")
a = pd.DataFrame(a)
y = a
sc = StandardScaler()


x_img = []

for nombre in os.listdir(r"kagglecatsanddogs_5340\train\train"):
    images = image.load_img(r"kagglecatsanddogs_5340\train\train\\" + nombre, target_size=(128, 128))
    
    x = image.img_to_array(images)
    x = tf.image.rgb_to_grayscale(x)
    #x = np.expand_dims(x, axis=0)
    x_img.append(x)

x = np.array(x_img).reshape(-1, 128*128)
x = sc.fit_transform(x)
model = MLPClassifier(hidden_layer_sizes=(128), max_iter=1000)
model.fit(x, y.values.ravel())

# Save the model to disk
filename = 'm_test5.pkl'
pickle.dump(model, open(filename, 'wb'))



