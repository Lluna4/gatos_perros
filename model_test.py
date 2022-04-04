import sklearn
import numpy as np
import pandas as pd
from keras_preprocessing import image
import tensorflow as tf
import os
import csv
from sklearn.neural_network import MLPClassifier
import pickle

model = pickle.load(open('m_test.pkl', 'rb'))

images = image.load_img("perro_redes.jpg", target_size=(28, 28))

x = image.img_to_array(images)
x = tf.image.rgb_to_grayscale(x)
x = np.array(x).reshape(1, 28*28)
a = model.predict(x)
if a == 0:
    print("cat")
else:
    print("dog")



