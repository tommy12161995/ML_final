import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import matplotlib.gridspec as gridspec
import os
import random

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
def sample_z(m, n):
    return np.random.uniform(-1, 1, size=[m, n])
    
def generator(z, is_training):
    def add_layer(inputs, in_size, out_size):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # outputs = tf.nn.leaky_relu(Wx_plus_b)
        return Wx_plus_b
    def add_deconv2d(layer_input, filters, k_size):
        u = tf.layers.conv2d_transpose(layer_input, filters, kernel_size = k_size, strides = (2, 2), padding = 'same')
        outputs = tf.nn.leaky_relu(u)
        return outputs
    with tf.variable_scope('Generator'):
        # z = tf.print(z, [tf.shape(z)], message = 'z')
        G_1 = tf.reshape(add_layer(z, 100, 8 * 16 * 512), [-1, 8, 16, 512])
        G_1 = tf.nn.leaky_relu(G_1)
        # G_1 = tf.layers.batch_normalization(G_1, momentum=0.9)
        # G_1 = tf.print(G_1, [tf.shape(G_1)], message = 'G_1')

        G_2 = add_deconv2d(G_1, 256, 5)
        G_2 = tf.layers.batch_normalization(G_2, momentum=0.9, training = is_training)
        G_2 = tf.nn.leaky_relu(G_2)
        # G_2 = tf.print(G_2, [tf.shape(G_2)], message = 'G_2')

        G_3 = add_deconv2d(G_2, 128, 5)
        G_3 = tf.layers.batch_normalization(G_3, momentum=0.9, training = is_training)
        G_3 = tf.nn.leaky_relu(G_3)
        # G_3 = tf.print(G_3, [tf.shape(G_3)], message = 'G_3')

        G_4 = add_deconv2d(G_3, 64, 5)
        G_4 = tf.layers.batch_normalization(G_4, momentum=0.9, training = is_training)
        G_4 = tf.nn.leaky_relu(G_4)
        # G_4 = tf.print(G_4, [tf.shape(G_4)], message = 'G_4')

        G_5 = add_deconv2d(G_4, 3, 5)
        G_5 = tf.nn.leaky_relu(G_5)
        # G_5 = tf.print(G_5, [tf.shape(G_5)], message = 'G_5')
        outputs = tf.nn.tanh(G_5)
        # outputs = tf.print(outputs, [tf.shape(outputs)], message = 'output')

    return outputs
    
def discriminator(x, reuse = False):
    def add_layer(inputs, in_size, out_size):
        inputs = tf.reshape(inputs, [-1, in_size])
        # inputs = tf.print(inputs, [tf.shape(inputs)])
        # inputs = tf.layers.flatten(inputs)
        # print(inputs.shape)
        # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # outputs = Wx_plus_b
        # outputs = tf.nn.leaky_relu(Wx_plus_b)
        outputs = tf.layers.dense(inputs, out_size)
        return outputs
    def add_conv2d(layer_input, filters, k_size):
        outputs = tf.layers.conv2d(layer_input, filters, kernel_size = k_size, strides = (2, 2), padding = 'same')
        outputs = lrelu(outputs)
        return outputs
    with tf.variable_scope('Discriminator') as scope:
        if reuse:
           scope.reuse_variables()
        # x = tf.print(x, [tf.shape(x)], message = 'x')
        D_1 = add_conv2d(x, 64, 5)
        # D_1 = tf.print(D_1, [tf.shape(D_1)], message = 'D_1')
        D_2 = add_conv2d(D_1, 128, 5)
        # D_2 = tf.print(D_2, [tf.shape(D_2)], message = 'D_2')
        D_3 = add_conv2d(D_2, 256, 5)
        # D_3 = tf.print(D_3, [tf.shape(D_3)], message = 'D_3')
        D_4 = add_conv2d(D_3, 512, 5)
        # D_4 = tf.print(D_4, [tf.shape(D_4)], message = 'D_4')
        D_5 = add_layer(D_4, 8 * 16 * 512, 2)
        outputs = tf.nn.softmax(D_5)

    return outputs, D_5
def load_batch(batch_size=1, img_res=(64, 64)):
    train_path = glob('./data/train_edge_data/*')
    label_path = glob('./data/train_edge_label/*')

    n_batches = int(len(train_path) / batch_size)

    for i in range(n_batches-1):
        train_batch = train_path[i*batch_size:(i+1)*batch_size]
        label_batch = label_path[i*batch_size:(i+1)*batch_size]
        imgs_A, imgs_B = [], []
        for train_img in train_batch:
            img_B = cv2.imread(train_img, cv2.IMREAD_COLOR)
            img_B = cv2.resize(img_B, img_res)
            imgs_B.append(img_B)
            
        for label_img in label_batch:
            img_A = cv2.imread(label_img, cv2.IMREAD_COLOR)
            img_A = cv2.resize(img_A, img_res)
            imgs_A.append(img_A)
        out_imgs = []
        for i in range(len(imgs_A)):
            concat_img = np.concatenate((imgs_B[i], imgs_A[i]), axis = 1)
            out_imgs.append(concat_img)
            # cv2.imshow('My Image', concat_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        out_imgs = (np.array(out_imgs) - 127.5) / 127.5 # -1 ~ 1

        yield out_imgs

def load_min_batch(batch_size=1, img_res=(128, 128)):
    train_path = glob('./edge_blur/train_blur_data/*')
    label_path = glob('./edge_blur/train_blur_label/*')
    batch_index = []
    count = 0
    num = len(train_path)

    while count < batch_size:
        index = random.randint(0, num - 1)
        if index in batch_index:
            continue
        batch_index.append(index)
        count += 1

    imgs_A, imgs_B = [], []
    for i in batch_index:
        img_B = cv2.imread(train_path[i], cv2.IMREAD_COLOR)
        img_B = cv2.resize(img_B, img_res)
        imgs_B.append(img_B)

        img_A = cv2.imread(label_path[i], cv2.IMREAD_COLOR)
        img_A = cv2.resize(img_A, img_res)
        imgs_A.append(img_A)

    out_imgs = []
    for i in range(len(imgs_A)):
        concat_img = np.concatenate((imgs_B[i], imgs_A[i]), axis = 1)
        out_imgs.append(concat_img)
        # cv2.imshow('My Image', concat_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    out_imgs = (np.array(out_imgs) - 127.5) / 127.5 # -1 ~ 1

    return out_imgs

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    # print(samples.shape)
    for i, sample in enumerate(samples):
        sample = (sample + 1.0) / 2
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(128, 256, 3), cmap='Greys_r')

    return fig
def lrelu(x, a = 0.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
if __name__ == '__main__':
    z = tf.placeholder(tf.float32, shape=[None, 100])
    X = tf.placeholder(tf.float32, shape=[None, 128, 256, 3])

    g_sample = generator(z, is_training = True)
    d_real, d_logit_real = discriminator(X, reuse=False)
    d_fake, d_logit_fake = discriminator(g_sample, reuse=True) 

    real_label = np.array([1, 0])
    fake_label = np.array([0, 1])

    D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = real_label, logits = d_logit_real))
    D_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = fake_label, logits = d_logit_fake))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = real_label, logits = d_logit_fake))

    learning_rate = 0.0002
    beta = 0.5
    discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("Discriminator")]
    D_solver = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta).minimize(
        D_loss, var_list=discrim_tvars)
    G_solver = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta).minimize(
        G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)generatorgenerator
    z_dim = 100
    i = 0
    batch_size = 64
    epochs = 200
    # for it in range(1000000):
    #     # if it % 1000 == 0:
    #     #     samples = sess.run(g_sample, feed_dict={z: sample_z(16, z_dim)})

    #     #     fig = plot(samples)
    #     #     plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    #     #     i += 1
    #     #     plt.close(fig)
        
    #     batch_i, label_batch = enumerate(load_batch(batch_size = batch_size))
    #     # print(label_batch.shape)

    #     _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: label_batch, z: sample_z(batch_size, z_dim)})
    #     _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: sample_z(batch_size, z_dim)})

    #     if it % 1000 == 0:
    #         print('Iter: {}'.format(it))
    #         print('D loss: {:.4}'. format(D_loss_curr))
    #         print('G_loss: {:.4}'.format(G_loss_curr))
    #         print()
    saver = tf.train.Saver()
    for it in range(20000):
        label_batch = load_min_batch(batch_size)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: label_batch, z: sample_z(batch_size, z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: sample_z(batch_size, z_dim)})
        if it % 100 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()
        
            samples = sess.run(g_sample, feed_dict={z: sample_z(4, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
            save_path = saver.save(sess, './model/model_' + str(it) + '.ckpt')
