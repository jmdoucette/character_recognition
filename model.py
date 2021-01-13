#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:32:06 2021

@author: jamesdoucette
"""
import numpy as np
import tensorflow as tf


from sklearn.metrics import classification_report
from emnist import extract_training_samples,extract_test_samples

from scikit_image.util import img_as_float,img_as_ubyte
from scikit_image import color
from  scikit_image.transform import resize

def convert_to_image(data):
    data = data / 255.0
    return np.repeat(data.reshape(len(data),28,28,1),3,axis=3)
    


def create_model(x_train,y_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
    
    model.fit(x_train,y_train)
    
    return model

def train(dataset):
    x_train,y_train = extract_training_samples(dataset)
    x_train = convert_to_image(x_train)
    return create_model(x_train,y_train)
    

def make_prediction(model,im):
    raw=model.predict(np.array([im]))
    return np.argmax(tf.nn.softmax(raw).numpy(),axis=1)[0]