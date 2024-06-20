import os
import tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tensorflow.__version__)