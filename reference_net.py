import keras
import numpy as np
import cv2
import os
import random
import shutil
import pandas as pd
import csv
import zipfile
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
import keras.backend as k
from sklearn.utils import shuffle
import io
from PIL import Image as pil_image
from keras_retinanet import layers
import keras.backend as k
import keras_retinanet
import tensorflow as tf
from tensorflow.python.client.session import InteractiveSession

dirname = os.path.dirname(__file__)

if not os.path.isdir(os.path.join(dirname, 'data')):
    archive = zipfile.ZipFile(os.path.join(dirname, 'Train&Validation.zip')) #Path to the shared data for training and validation
    for file in archive.namelist():
         archive.extract(file, './data') #Extract the data

fold_num=1 #Select Fold Number

#Here we set the data generators for applying data augmentation methods
train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=0.05,rotation_range=360,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.05)
test_datagen = ImageDataGenerator()
train_df =pd.read_csv(os.path.join(dirname, 'csv/train{}.csv').format(fold_num)) #raed train csv file
validation_df = pd.read_csv(os.path.join(dirname, 'csv/validation{}.csv').format(fold_num)) #raed validation csv file (Validation in the training process)
train_df = shuffle(train_df) #Shuffle the train data
test_df = pd.read_csv(os.path.join(dirname, 'csv/test{}.csv').format(fold_num))#raed test csv file (For evaluating the final version of the trained network)

shape=(512,512,1) #shape of the dataset images (in TIFF format)

# Set by me to test GPU
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#Create the generators
train_generator = train_datagen.flow_from_dataframe(
      dataframe=train_df,
      directory='data',
      x_col="filename",
      y_col="class",
      target_size=shape[:2],
      batch_size=14,
      class_mode='categorical',color_mode="grayscale",shuffle=True)
validation_generator = test_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory='data',
        x_col="filename",
        y_col="class",
        target_size=shape[:2],
        batch_size=10,
        class_mode='categorical',color_mode="grayscale",shuffle=True)
test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='data',
        x_col="filename",
        y_col="class",
        target_size=shape[:2],
        batch_size=10,
        class_mode='categorical',color_mode="grayscale",shuffle=True)

k.clear_session() #Clear keras backend
try:
  os.mkdir('models') #create folder for saving the trained networks
except:
  pass
full_name='ResNet50V2-FPN-fold{}'.format(fold_num)
classes_number=2 #Number of classes (normal and COVID-19)
input_tensor=Input(shape=shape)
weight_model = ResNet50V2(weights='imagenet', include_top=False) #Load ResNet50V2 ImageNet pre-trained weights
weight_model.save_weights('weights.h5') #Save the weights
base_model = ResNet50V2(weights=None, include_top=False, input_tensor=input_tensor) #Load the ResNet50V2 model without weights
base_model.load_weights('weights.h5',skip_mismatch=True, by_name=True) #Load the ImageNet weights on the ResNet50V2 model except the first layer(because the first layer has one channel in our case)

#Create Feature Pyramid Network (FPN)
# We used some help for writing the Pyramid from the written code on https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/models/retinanet.py
feature_size=256 #Set the feature channels of the FPN
layer_names = ["conv4_block1_preact_relu", "conv5_block1_preact_relu", "post_relu"] #Layers of ResNet50V2 with different scale features
layer_outputs = [base_model.get_layer(name).output for name in layer_names]
C3, C4, C5=layer_outputs #Features of different scales, extracted from ResNet50V2
P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

# Concatenate P5 elementwise to C4
P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
P4           = keras.layers.Concatenate(axis=3)([P5_upsampled, P4])
P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, name='P4')(P4)

# Concatenate P4 elementwise to C3
P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
P3 = keras.layers.Concatenate(axis=3)([P4_upsampled, P3])
P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, name='P3')(P3)

# "P6 is obtained via a 3x3 stride-2 conv on C5"
P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

# "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

# Run classification for each of the generated features from the pyramid
feature1 = Flatten()(P3)
dp1 = Dropout(0.5)(feature1)
preds1 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp1)
#################################################################
feature2 = Flatten()(P4)
dp2 = Dropout(0.5)(feature2)
preds2 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp2)
#################################################################
feature3 = Flatten()(P5)
dp3= Dropout(0.5)(feature3)
preds3 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp3)
#################################################################
feature4 = Flatten()(P6)
dp4 = Dropout(0.5)(feature4)
preds4 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp4)
#################################################################
feature5 = Flatten()(P7)
dp5 = Dropout(0.5)(feature5)
preds5 = Dense(2, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(dp5)
#################################################################
concat=keras.layers.Concatenate(axis=1)([preds1,preds2,preds3,preds4,preds5]) #Concatenate the predictions(Classification results) of each of the pyramid features
out=keras.layers.Dense(2,activation='softmax',kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(concat) #Final Classification

model = Model(inputs=base_model.input, outputs=out) #Create the Training Model
#######################################################
for layer in model.layers:
  layer.trainable = True
model.compile(optimizer=optimizers.Nadam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
filepath="models/%s-{epoch:02d}-{val_accuracy:.4f}.hdf5"%full_name  # Path to save the trained models
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max') #creating checkpoint to save the best validation accuracy
callbacks_list = [checkpoint]

model.fit(train_generator, epochs=20,validation_data=validation_generator,shuffle=True,callbacks=callbacks_list) #start training

#Model Evaluation
trained_models=[]
for r,d,f in os.walk('models'): #Take the path to the trained nets
  for file in f:
    if '.hdf5' in file:
      trained_models.append(os.path.join(r,file))

reports=[]
for trn_model in trained_models: #evaluate the network on each trained net
  k.clear_session()
  custom_object={'UpsampleLike': keras_retinanet.layers._misc.UpsampleLike}
  netpath=trn_model
  model_name=trn_model
  fold_num=trn_model[trn_model.index('fold')+4] #find the fold number
  net=keras.models.load_model(netpath, custom_objects=custom_object) #load model

  covid_label= test_generator.class_indices['covid'] #get the index of COVID-19 class
  normal_label= test_generator.class_indices['normal']  #get the index of normal class
  tp=0 #True Positives
  fp=0 #False Positives
  anum=0 #All the images numbers
  ###########
  ctp=0 #Correct classified COVID-19 cases
  cfp=0 #Wrong classified COVID-19 cases
  cfn=0 #Not classified COVID-19 cases
  ctn=0 #Correctly not classified COVID-19 cases
  cnum=0 #Number of COVID cases
  ################
  ntp=0 #Correct classified normal cases
  nfp=0 #Wrong classified normal cases
  nfn=0 #Not classified normal cases
  ntn=0 #Correctly not classified normal cases
  nnum=0 #Number of normal cases
  for num,img_name in enumerate(test_generator.filenames): #load image
    gt_ind=test_generator.classes[num] #get the loaded image class index
    img=cv2.imread(os.path.join('data',img_name),cv2.IMREAD_UNCHANGED) #load image
    pred_ind=np.argmax(net.predict(np.expand_dims(np.expand_dims(img,axis=0),axis=3))[0]) #get the predicted class index
    anum+=1 #count the number of images
    if gt_ind==covid_label:
      cnum+=1
      if pred_ind==covid_label:
        tp+=1
        ctp+=1
        ntn+=1
      else:
        fp+=1
        nfp+=1
        cfn+=1
    elif gt_ind==normal_label:
      nnum+=1
      if pred_ind==normal_label:
        ctn+=1
        ntp+=1
        tp+=1
      else:
        cfp+=1
        nfn+=1
        fp+=1

  overall_acc=tp/(tp+fp) #overall accuracy
  cacc=(ctp+ctn)/(ctp+ctn+cfp+cfn) #covid accurayc
  nacc=(ntp+ntn)/(ntp+ntn+nfp+nfn) #normal accuracy
  csens=ctp/(ctp+cfn) #covid sensitivity
  nsens=ntp/(ntp+nfn) #normal sensitivity
  cspec=ctn/(ctn+cfp) #covid specificity
  nspec=ntn/(ntn+nfp) #normal specificity
  cprec=ctp/(ctp+cfp) #covid precision
  nprec=ntp/(ntp+nfp) #normal precision

  reports.append([model_name,fold_num,tp,fp,ctp,cfn,cfp,ntp,nfn,nfp,overall_acc,cacc,nacc,csens,nsens,cspec,nspec,cprec,nprec])


  print(model_name)
  print('tp: ',tp,'fp: ',fp)

with open('FPN.csv', mode='w',newline='') as csv_file:
    csvwriter = csv.writer(csv_file, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['model_name','fold_num','tp','fp','ctp','cfn','cfp','ntp','nfn','nfp','overall_acc','cacc','nacc','csens','nsens','cspec','nspec','cprec','nprec'])
    for row in reports:
        csvwriter.writerow(row)