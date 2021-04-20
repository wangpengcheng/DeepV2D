
from geometry.intrinsics import *
from geometry.projective_ops import *
from geometry.se3 import *
import tensorflow as tf
if __name__ == '__main__':
    # a = tf.Variable(tf.ones([100,640,480,3]), name="v2")
    # b = tf.Variable(tf.ones([100,4,4]), name="v2")
    #c = rescale_depths_and_intrinsics(a,b)
    #//coords =  project(a,b)
    #coords1= backproject(a,b)
    a = tf.Variable(tf.ones([3,3]), name="v1")
    b = tf.Variable(tf.ones([1,3]), name="v2")
    

    c = matdotv(a,b)
    print(a)
    print(c)
