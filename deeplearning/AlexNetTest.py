import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
import numpy
import random
import torch
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import layers
from tensorflow.python.client import device_lib
random.seed(100)
numpy.random.seed(100)
tf.random.set_seed(100)

# Set GPU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = ("cuda" if torch.cuda.is_available() else "cpu") # device 정의
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1)

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))
validation_ds = tf.data.Dataset.from_tensor_slices((x_val,y_val))

def process_images(image, label):
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds = train_ds.batch(batch_size=128, drop_remainder=True)
test_ds = test_ds.batch(batch_size=128, drop_remainder=True)
val_ds = validation_ds.batch(batch_size=128, drop_remainder=True)

train_ds=train_ds.map(process_images)
test_ds=test_ds.map(process_images)
val_ds=val_ds.map(process_images)
# Reshape
# input shape: (data수, image_width, image_height, channel_num)
#image_w, image_h = 227, 227
#x_train = x_train.resize(image_w, image_h, 3)
#x_valid = x_valid.resize(image_w, image_h, 3)
#x_test = x_test.resize(image_w, image_h, 3)

model = keras.Sequential()
model.add(layers.Conv2D(96, (11,11), strides=4, padding='valid', activation='relu', input_shape=(227,227,3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((3,3), strides=2, padding='valid'))


model.add(layers.Conv2D(256, (5,5), strides=1, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2), strides=2, padding='valid'))


model.add(layers.Conv2D(384, (3,3), strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(384, (3,3), strides=1, padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3,3), strides=1, padding='same', activation='relu'))
model.add(layers.MaxPool2D((3,3), strides=2, padding='valid'))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))


# Model Compile
model.compile(loss = 'sparse_categorical_crossentropy', 
            optimizer = 'adam',
            metrics = ['accuracy'])
model.build()
model.summary()
# Model Fit
history = model.fit(train_ds, epochs=10, validation_data=val_ds, validation_freq=1)

# Model test
model.evaluate(y_test)