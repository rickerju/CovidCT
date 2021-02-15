"""
Title: Transfer learning & fine-tuning
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/15
Last modified: 2020/05/12
Description: Complete guide to transfer learning & fine-tuning in Keras.
"""
import os

from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client.session import InteractiveSession

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
train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=0.05,rotation_range=360,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.05)
test_datagen = ImageDataGenerator()
train_df = pd.read_csv(os.path.join(dirname, 'fold-csv/train.csv')) #raed train csv file
validation_df = pd.read_csv(os.path.join(dirname, 'fold-csv/validation.csv'))
train_generator = train_datagen.flow_from_dataframe(
      dataframe=train_df,
      directory='fold-data',
      x_col="filename",
      y_col="class",
      target_size=shape[:2],
      batch_size=14,
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
# x = keras.layers.Dense(2048)(x)
outputs = keras.layers.Dense(2)(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

epochs = 20
model.summary()
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# """
# ## Do a round of fine-tuning of the entire model
# Finally, let's unfreeze the base model and train the entire model end-to-end with a low
#  learning rate.
# Importantly, although the base model becomes trainable, it is still running in
# inference mode since we passed `training=False` when calling it when we built the
# model. This means that the batch normalization layers inside won't update their batch
# statistics. If they did, they would wreck havoc on the representations learned by the
#  model so far.
# """
#
# # Unfreeze the base_model. Note that it keeps running in inference mode
# # since we passed `training=False` when calling it. This means that
# # the batchnorm layers will not update their batch statistics.
# # This prevents the batchnorm layers from undoing all the training
# # we've done so far.
# base_model.trainable = True
# model.summary()
#
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
#     loss=keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=[keras.metrics.BinaryAccuracy()],
# )
#
# epochs = 10
# model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
#
# """
# After 10 epochs, fine-tuning gains us a nice improvement here.
# """