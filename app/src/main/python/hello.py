from java import jclass
from tensorflow.python.client import device_lib

import tensorflow as tf

def getgpu():
    print(device_lib.list_local_devices())

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

def greet(name):
    print("--- hello,%s ---" % name)

def add(a,b):
    return a + b

def sub(count,a=0,b=0,c=0):
    return count - a - b -c

def get_list(a,b,c,d):
    return [a,b,c,d]

def print_list(data):
    print(type(data))
    # 遍歷Java的ArrayList對象
    for i in range(data.size()):
        print(data.get(i))
