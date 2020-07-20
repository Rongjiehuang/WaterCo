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
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
import yaml
import h5py
from PIL import Image
from net.fcn import fcn_net


UNIFIED_HEIGHT = 256
UNIFIED_WIDTH = 256
BATCH_SIZE = 5
TRAIN_NUM = 700
VALID_NUM = 300
C1 = 0.4680
C2 = 0.2745
C3 = 0.2576

def generate_arrays(path,batch_size):
    # 获取总长度
    file_name=os.listdir(path+"/input/")
    j = 0
    file_num = len(file_name)
    img_name=os.listdir(path+"/input/"+file_name[j]+'/')
    i = 0
    img_num = len(img_name)

    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            # 从文件中读取输入图像
            input_img = Image.open(path+"/input/"+file_name[j]+'/'+img_name[i])
            input_img = input_img.resize((UNIFIED_WIDTH,UNIFIED_HEIGHT))
            # input_img.show()
            input_img = np.array(input_img)
            input_img = input_img/255
            X_train.append(input_img)

            # 从文件中读取输出图像
            output_img = Image.open(path+"/output/"+file_name[j]+'/'+img_name[i])
            output_img = output_img.resize((UNIFIED_WIDTH,UNIFIED_HEIGHT))
            # output_img.show()
            output_img = np.array(output_img)
            output_img = output_img/255
            Y_train.append(output_img)

            # 读完一个周期后重新开始
            i = i+1
            if i == img_num:
                j = (j+1)%file_num
                img_name = os.listdir(path+"/input/"+file_name[j]+'/')
                i = 0
                img_num = len(img_name)

        yield (np.array(X_train),np.array(Y_train))

def img_fac(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_H,img_L,img_S=cv2.split(img_hsv)
    chroma_mean,chroma_stddev=cv2.meanStdDev(img_S)
    min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(img_L)
    lightness_contrast=max_val-min_val
    return chroma_stddev*255,lightness_contrast*255,chroma_mean*255

def loss(y_true, y_pred):
    loss = keras.losses.mean_absolute_error(y_true,y_pred)
    return loss

def metrics(y_true, y_pred):
    # img_true=np.float32(y_true)
    # chroma_stddev,lightness_contrast,chroma_mean=img_fac(img_true)
    # UCIQE_true=C1*chroma_stddev+C2*lightness_contrast+C3*chroma_mean
    # img_pred=np.float32(y_pred)
    # chroma_stddev,lightness_contrast,chroma_mean=img_fac(img_pred)
    # UCIQE_pred=C1*chroma_stddev+C2*lightness_contrast+C3*chroma_mean
    metrics = tf.losses.mean_squared_error(y_true,y_pred)
    return metrics

if __name__ == "__main__":
    train_path="F:/我的资源/data/train"
    valid_path="F:/我的资源/data/valid"
    # weights_path="D:/jupyter/FCN/logs/ep023-loss0.026-val_loss0.193.h5"
    log_dir = "logs2/"

    # 获取model
    model = fcn_net(BATCH_SIZE)
    model.summary()

    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=1
                                )

    # 学习率下降的方式，val_loss3次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )
    # 曼哈顿 自适应矩估计 水下彩色图像评估指标UCIQE
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-3),
            metrics = [metrics])

    # model.load_weights(weights_path,by_name=True)
    # 开始训练
    history = model.fit_generator(generate_arrays(train_path, BATCH_SIZE),
            steps_per_epoch=max(1, 140),
            validation_data=generate_arrays(valid_path, BATCH_SIZE),
            validation_steps=max(1, 60),
            epochs=100,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr, early_stopping])

    model.save_weights(log_dir+'weights20200720.h5')