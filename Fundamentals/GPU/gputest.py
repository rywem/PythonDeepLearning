# -*- coding: utf-8 -*-


# Installation Guide: https://www.tensorflow.org/install/gpu#windows_setup


# Code source: https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
import tensorflow as tf

from tensorflow.python.client import device_lib


def useGPU():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    
    with tf.Session() as sess:
        print (sess.run(c))
        
def logDevicePlacement():    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run()
    
logDevicePlacement();