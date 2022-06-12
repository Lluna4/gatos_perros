import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
import os
from ann_visualizer.visualize import ann_viz
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'



clf = tf.keras.models.load_model('weights-improvement-22-0.93')
ann_viz(clf, view=True, title="aaaa")