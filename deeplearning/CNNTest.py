import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Model, layers

random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Load data
(x_train_val, y_train_val),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size= 0.2, shuffle=True, stratify = y_train_val, random_state=34)

print(x_train.shape) #=> (48000, 28, 28) 
print(y_train.shape) #=> (48000,)
print(x_train[0].shape) #=> (28, 28) 이미지 한장 크기

print(x_valid.shape) #=> (12000, 28, 28)
print(x_test.shape) #=> (10000, 28, 28)

plt.imshow(x_train[1])
plt.show()

# Data Preprocessing
num_classes = len(set(y_train)) # MNIST는 10개의 class를 가진다.
print(num_classes) #=> 10
# Scale to [0,1]
x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape
# input shape: (data수, image_width, image_height, channel_num)
image_w, image_h = x_train.shape[1], x_train.shape[2]
print(x_train.shape[1], x_train.shape[2]) 
print(x_train.shape, x_valid.shape, x_test.shape) 

x_train = x_train.reshape(-1, image_w, image_h, 1)
x_valid = x_valid.reshape(-1, image_w, image_h, 1)
x_test = x_test.reshape(-1, image_w, image_h, 1)

print(x_train.shape, x_valid.shape, x_test.shape) 
#=> (48000, 28, 28, 1) (12000, 28, 28, 1) (10000, 28, 28, 1)
print(y_train.shape, y_valid.shape, y_test.shape) 
#=> (48000,) (12000,) (10000,)

# layers.Conv2D( 생성할 커널수, 커널사이즈, 커널이동거리, 패딩방법지정, 활성화함수지정, 입력데이터 차원정보(첫레이어일때만) )
# layers.MaxPool2D( 커널사이즈, 커널이동거리, 패딩방법지정 )
# layers.Dense( 노드수, 활성화함수지정 )

class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=(28,28,1))
        self.maxpool1 = layers.MaxPool2D((3,3), strides=2, padding='same')
        
        self.conv2 = layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu')
        self.maxpool2 = layers.MaxPool2D((3,3), strides=2, padding='same')
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

        

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

cnn = CNN()

# Model Compile
cnn.compile(loss = 'sparse_categorical_crossentropy', 
            optimizer = 'SGD',
            metrics = ['accuracy'])

# Model Fit
history = cnn.fit(x_train, y_train, batch_size=128, # 1epoch에 모든 데이터를 한번에 돌리는게 아닌, batch로 나누어 돌림
                  epochs=10, # 반복학습 횟수                 
                  validation_data=(x_test, y_test)) # 검증셋

# Model test
cnn.evaluate(x_test, y_test)
#출처: https://csshark.tistory.com/132