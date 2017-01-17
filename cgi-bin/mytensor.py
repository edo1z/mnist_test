import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

class MyTensor:
    H = 625
    BATCH_SIZE = 100
    DROP_OUT_RATE = 0.5

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.t = tf.placeholder(tf.float32, [None, 10])        
        self.w1 = tf.Variable(tf.random_normal([784, self.H], mean=0.0, stddev=0.05))
        self.b1 = tf.Variable(tf.zeros([self.H]))
        self.w2 = tf.Variable(tf.random_normal([self.H, self.H], mean=0.0, stddev=0.05))
        self.b2 = tf.Variable(tf.zeros([self.H]))
        self.w3 = tf.Variable(tf.random_normal([self.H, 10], mean=0.0, stddev=0.05))
        self.b3 = tf.Variable(tf.zeros([10]))

        self.a1 = tf.sigmoid(tf.matmul(self.x, self.w1) + self.b1)
        self.a2 = tf.sigmoid(tf.matmul(self.a1, self.w2) + self.b2)
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop = tf.nn.dropout(self.a2, self.keep_prob)
        self.y = tf.nn.relu(tf.matmul(self.drop, self.w3) + self.b3)
        self.loss = tf.nn.l2_loss(self.y - self.t) / self.BATCH_SIZE

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
        self.saver = tf.train.Saver()
        if not self.ckpt():
            self.train()

    def ckpt(self):
        return tf.train.get_checkpoint_state('.\cgi-bin\ckpt')        

    def predict(self, img):
        with tf.Session() as sess:
            self.saver.restore(sess, self.ckpt().model_checkpoint_path)
            result = sess.run(self.y, feed_dict={self.x: img, self.keep_prob:1.0})
            return int(np.argmax(result))

    def train(self):
        mnist = input_data.read_data_sets('.\cgi-bin\mnist', one_hot=True, dtype=tf.uint8)        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(20000):
                batch_x, batch_t = mnist.train.next_batch(100)
                sess.run(self.train_step, feed_dict={self.x: batch_x, self.t: batch_t, self.keep_prob:(1-self.DROP_OUT_RATE)})
            self.saver.save(sess, '.\cgi-bin\ckpt\model.ckpt')
    