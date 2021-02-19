import tensorflow as tf
import numpy as np

'''对 constant'''
c = tf.constant([[[1, 2, 3]]])
print(c.shape.as_list())
c_squeeze = tf.squeeze(c)  # 对 constant 用 squeeze
print(c_squeeze.shape.as_list())  # 可以

'''对 placeholder'''
ph = tf.placeholder("float32", [None, None, None, 3])
print(ph.shape.as_list())  # squeeze 前，可以

ph_squeeze = tf.squeeze(ph)  # 对 placeholder 用 squeeze
# print(ph_squeeze.shape.as_list())  # squeeze 后，报错
print(ph.shape.as_list())

a = ph.shape.as_list()
for c in a :
    print(c)

print(a[-1])
ph_reshape = tf.reshape(ph, a)  # 换成 reshape
print(ph_reshape.shape.as_list())  # 可以

'''老方法'''
with tf.Session() as sess:
    ph_squeeze_shape = tf.shape(ph_squeeze)
    print("ph_squeeze shape:", sess.run(ph_squeeze_shape,feed_dict={ph: np.zeros([1, 1, 1, 3])}))
