from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import os
import math
from PIL import Image
import numpy as np
import time
    
# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        image_raw = Image.open(FLAGS.target_dir)
        if FLAGS.texture_shape == [-1,-1]:
            image_raw = check_size(image_raw)
        else:
            image_raw = image_raw.resize( (FLAGS.texture_shape[0], FLAGS.texture_shape[1]) )
        if image_raw.mode is not 'RGB':
            image_raw = image_raw.convert('RGB')
        image_raw = np.asarray(image_raw)/255
        targets = preprocess(image_raw)  
        samples = np.expand_dims(targets, axis = 0) 
    return samples

def generator(FLAGS, target, init = None):
    if init is not None:
        var = tf.Variable(init + tf.random_normal(tf.shape(init), 0, FLAGS.stddev))  
    else:
        if FLAGS.texture_shape == [-1,-1]:
            shape = [1, target.shape[1],target.shape[2],3]
        else:
            shape = [1, FLAGS.texture_shape[0],FLAGS.texture_shape[1],3]
        var = tf.get_variable('gen_img',shape = shape, \
                  initializer = tf.random_normal_initializer(0,0.5), \
                               dtype=tf.float32,trainable=True, collections=None)   
    return tf.tanh(var)

def Synthesis(initials, targets, upsampling, FLAGS):
    with tf.variable_scope('generator'):
        if initials is None:
            pass
        else:
            w, h = [ initials.shape[1], initials.shape[2] ]
            try:
                initials = tf.constant(initials)
            except:
                pass
            if upsampling == True:
                    initials = tf.image.resize_bicubic(initials, [2*int(w), 2*int(h)])
        gen_output = generator(FLAGS, targets, initials)

        # Calculating the generator loss
    with tf.name_scope('generator_loss'):   
        with tf.name_scope('tv_loss'):
            tv_loss = total_variation_loss(gen_output)

        with tf.name_scope('style_loss'):
            _, vgg_gen_output = vgg_19(gen_output,is_training=False, reuse = False)
            _, vgg_tar_output = vgg_19(targets,   is_training=False, reuse = True)
            style_layer_list = get_layer_list(FLAGS.top_style_layer,False)
            sl = tf.zeros([])
            ratio_list=[100.0, 1.0, 0.1, 0.0001, 1.0, 100.0]
            for i in range(len(style_layer_list)):
                tar_layer = style_layer_list[i]
                target_layer = get_layer_scope(tar_layer)
                gen_feature = vgg_gen_output[target_layer]
                tar_feature = vgg_tar_output[target_layer]
                diff = tf.square(gram(gen_feature) - gram(tar_feature))
                sl = sl + tf.reduce_mean(tf.reduce_sum(diff, axis=0)) * ratio_list[i] 
            style_loss = sl
                
        gen_loss = style_loss + FLAGS.W_tv * tv_loss
        gen_loss = 1e6 * gen_loss

    with tf.name_scope('generator_train'):
        gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            gen_loss, var_list = gen_tvars, method='L-BFGS-B',
            options = {'maxiter': FLAGS.max_iter, 'disp': FLAGS.print_loss})

    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)

    # Start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def print_loss(gl, sl, tvl):
        if FLAGS.print_loss is True:
            print('gen_loss : %s' % gl )
            print('style_loss : %s' % sl )
            print('tv_loss : %s' % tvl )
    
    init_op = tf.global_variables_initializer()   
    with tf.Session(config = config) as sess:
        sess.run(init_op)
        vgg_restore.restore(sess, FLAGS.vgg_ckpt)
        print('Under Synthesizing ...')
        #start = time.time()
        optimizer.minimize(sess, loss_callback = print_loss,
                         fetches = [gen_loss, style_loss, tv_loss])

        return gen_output.eval(), style_loss.eval()
