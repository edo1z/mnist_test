import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def interface():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    W_conv1 = _weight_variable([5, 5, 1, 32])
    b_conv1 = _bias_variable([32])
    h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = _weight_variable([5, 5, 32, 64])
    b_conv2 = _bias_variable([64])
    h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_fc1 = _weight_variable([7 * 7 * 64, 1024])
    b_fc1 = _bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = _weight_variable([1024, 10])
    b_fc2 = _bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    class CNNModel():
        pass
    model = CNNModel()
    model.x = x
    model.y_ = y_
    model.keep_prob = keep_prob
    model.y_conv = y_conv
    model.train_step = train_step
    model.accuracy = accuracy
    model.saver = saver
    return model

def predict(img):
    ckpt = tf.train.get_checkpoint_state('./cgi-bin/ckpt')
    if not ckpt: return False
    m = interface()
    with tf.Session() as sess:
        m.saver.restore(sess, ckpt.model_checkpoint_path)
        result = sess.run(m.y_conv, feed_dict={m.x: img, m.keep_prob:1.0})
        return int(np.argmax(result))

def train():
    if tf.train.get_checkpoint_state('./ckpt'):
        print('train ok')
        return
    mnist = input_data.read_data_sets('./mnist', one_hot=True, dtype=tf.float32)
    m = interface()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(100)
            m.train_step.run(feed_dict={m.x: batch[0], m.y_: batch[1], m.keep_prob: 0.5})
            if i % 100 == 0:
                train_accuracy = m.accuracy.eval(feed_dict={m.x:batch[0], m.y_: batch[1], m.keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
        m.saver.save(sess, './ckpt/model.ckpt')
    print('train ok')

if __name__ == '__main__':
    train()