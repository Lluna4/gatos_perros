import tensorflow as tf


model = tf.keras.models.load_model('weights-improvement-22-0.93')

# Check its architecture
model.summary()