import tensorflow as tf
from keras.models import load_model

model_path = 'model.h5'
model = load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

lite_model = converter.convert()

