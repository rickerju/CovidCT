import os

from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client.session import InteractiveSession

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.python.keras import Input

dirname = os.path.dirname(__file__)

# Set by me to test GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Create the generators
# size = (150, 150)
# shape = (150, 150, 3)
# train_ds, validation_ds, test_ds = tfds.load(
#     "cats_vs_dogs",
#     # Reserve 10% for validation and 10% for test
#     split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
#     as_supervised=True,  # Include labels
# )
# train_generator = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# validation_generator = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
# test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))
#
# batch_size = 32
#
# train_generator = train_generator.cache().batch(batch_size).prefetch(buffer_size=10)
# validation_generator = validation_generator.cache().batch(batch_size).prefetch(buffer_size=10)
# test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

shape = (512, 512, 1)
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_df = pd.read_csv(os.path.join(dirname, 'fold-csv/train.csv')) #raed train csv file
validation_df = pd.read_csv(os.path.join(dirname, 'fold-csv/validation.csv'))
train_generator = train_datagen.flow_from_dataframe(
      dataframe=train_df,
      directory='fold-data',
      x_col="filename",
      y_col="class",
      target_size=shape[:2],
      batch_size=10,
      class_mode='categorical',color_mode="grayscale",shuffle=True)
validation_generator = test_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory='fold-data',
        x_col="filename",
        y_col="class",
        target_size=shape[:2],
        batch_size=10,
        class_mode='categorical',color_mode="grayscale",shuffle=True)

# input_tensor=Input(shape=shape)
weight_model = ResNet50V2(weights='imagenet', include_top=False) #Load ResNet50V2 ImageNet pre-trained weights
weight_model.save_weights('weights.h5') #Save the weights
base_model = ResNet50V2(weights=None, include_top=False, input_shape=shape) #Load the ResNet50V2 model without weights
base_model.load_weights('weights.h5',skip_mismatch=True, by_name=True) #Load the ImageNet weights on the ResNet50V2 model except the first layer(because the first layer has one channel in our case)

base_model.trainable = False

inputs = keras.Input(shape=shape)

norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 1)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(inputs)
norm_layer.set_weights([mean, var])

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(2)(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

epochs = 100
model.summary()
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

print(history.history.keys())

plt.figure(1)
plt.xticks(np.arange(0, 21, 1.0))

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig("fig1.png")
plt.show()
