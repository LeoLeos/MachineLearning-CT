#! usr/bin/python
# -*- coding:utf-8 -*-
# @Author: winston he
# @File: test.py
# @Time: 2021-01-20 09:50
# @Email: winston.wz.he@gmail.com
# @Desc:


import numpy as np  # matrix tools
import matplotlib.pyplot as plt  # for basic plots
import seaborn as sns  # for nicer plots
import pandas as pd
from glob import glob
import re
from skimage.io import imread
import keras
import os
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, MaxPooling2D


datasetPath = '/Users/winstonhe/Desktop/LeoLucky/Python_Project/KaggleData/CTDataset'
overviewPath = '/Users/winstonhe/Desktop/LeoLucky/Python_Project/KaggleData/CTDataset/overview.csv'

# 获取文件名
def fileInfo(path):
    for dirname, _, filenames in os.walk(datasetPath):
        print(dirname,"______")
        for filename in filenames:
            print(os.path.join(dirname, filename))

# 打印数据集信息
def dataMeta(data):
    print(data.info())
    print(data.head())

# 读取数据
def getData(path):
    return pd.read_csv(path)

# visualize Age
def visualizeAge(data):
    plt.figure(figsize=(10, 5))
    sns.distplot(data['Age'])

    g = sns.FacetGrid(overview, hue="Contrast", size=6, legend_out=True)
    g = g.map(sns.distplot, "Age").add_legend()

    plt.show()

# 返回某一类型文件的路径名的列表
def pathList(datasetPath):
    all_images_list = glob(os.path.join(datasetPath, 'tiff_images', '*.tif'))
    return all_images_list

# 展示图片
def showCT(imagesPathList):
    f, ax = plt.subplots(10, 10, figsize=(10, 10))
    for index, i in enumerate(imagesPathList):
        ax[int(index/10), index%10].matshow(imread(i), cmap='gray')
    plt.show()

# 设置label
def setLabel(imagesPathList):
    check_contrast = re.compile(r'ID_([\d]+)_AGE_([\d]+)+_CONTRAST_([\d]+)_CT.tif')
    label = []
    id_list = []
    age = []
    for image in imagesPathList:
        id_list.append(check_contrast.findall(image)[0][0])
        age.append(check_contrast.findall(image)[0][1])
        label.append(check_contrast.findall(image)[0][2])

    return label, id_list

# 设置shape
def putShape(imagesPathList):
    jimread = lambda x: np.expand_dims(imread(x)[::2, ::2], 0)
    images = np.stack([jimread(i) for i in imagesPathList], 0)
    label, id_list = setLabel(imagesPathList)
    label_list = pd.DataFrame(label, id_list)
    X_train, X_test, y_train, y_test = train_test_split(images, label_list, test_size=0.1, random_state=0)
    n_train, depth, width, height = X_train.shape
    n_test, _, _, _ = X_test.shape
    input_shape = (width, height, depth)

    # 归一化
    input_train = X_train.reshape((n_train, width, height, depth))
    input_train = input_train / np.max(input_train)
    # 自己加的
    input_test = X_test.reshape((n_test, width, height, depth))
    input_test = input_test / np.max(input_test)

    # 设置类别
    output_train = keras.utils.to_categorical(y_train, 2)
    output_test = keras.utils.to_categorical(y_test, 2)

    # 设置参数
    batch_size = 20
    epochs = 40

    # 构建网络
    model2 = Sequential()
    model2.add(Conv2D(50, (5, 5), activation='relu', input_shape=input_shape))
    # 32개의 4x4 Filter 를 이용하여 Convolutional Network생성
    model2.add(MaxPooling2D(pool_size=(3, 3)))  # 3x3 Maxpooling
    model2.add(Conv2D(30, (4, 4), activation='relu', input_shape=input_shape))
    model2.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 Maxpooling
    model2.add(Flatten())  # 쭉풀어서 Fully Connected Neural Network를 만든다.
    model2.add(Dense(2, activation='softmax'))

    # 显示训练模型摘要
    print(model2.summary())

    # 整合模型
    model2.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # 开始训练
    history = model2.fit(input_train, output_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(input_test, output_test))

    # 评估结果
    score = model2.evaluate(input_test, output_test, verbose=0)
    print(score)

    # 可视化训练过程
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.show()



# main
# fileInfo(datasetPath)
# 读取数据
# overview = getData(overviewPath)
#
# visualizeAge(overview)
pathlist = pathList(datasetPath)
# showCT(pathlist)
setLabel(pathlist)
putShape(pathlist)
