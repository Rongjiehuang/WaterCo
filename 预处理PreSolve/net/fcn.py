import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


UNIFIED_HEIGHT = 256
UNIFIED_WIDTH = 256

def fcn_net(batch_size):
    input_data = KL.Input(shape=(UNIFIED_HEIGHT,UNIFIED_WIDTH,3))
    # encode
    x=(KL.Convolution2D(
        batch_input_shape=(batch_size, UNIFIED_HEIGHT, UNIFIED_WIDTH, 3),
        filters=32, kernel_size=3, strides=1, activation='relu', padding='same',))(input_data)
    x=(KL.MaxPooling2D(
        pool_size=(2, 2), strides=(2,2), padding='same'))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.MaxPooling2D(
        pool_size=(2, 2), strides=(2,2), padding='same'))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.MaxPooling2D(
        pool_size=(2, 2), strides=(2,2), padding='same'))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.MaxPooling2D(
        pool_size=(2, 2), strides=(2,2), padding='same'))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.MaxPooling2D(
        pool_size=(2, 2), strides=(2,2), padding='same'))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)

    #decode
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.UpSampling2D(size=(2,2)))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.UpSampling2D(size=(2,2)))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.UpSampling2D(size=(2,2)))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.UpSampling2D(size=(2,2)))(x)

    x=(KL.Convolution2D(
        filters=64, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    x=(KL.UpSampling2D(size=(2,2)))(x)

    x=(KL.Convolution2D(
        filters=32, kernel_size=3, strides=1, activation='relu', padding='same',))(x)
    
    output_data=(KL.Convolution2DTranspose(
        output_shape=(batch_size, UNIFIED_HEIGHT, UNIFIED_WIDTH, 3),
        filters=3, kernel_size=3, strides=1, activation='linear', padding='same'))(x)

    model=KM.Model(input_data,output_data)

    return model