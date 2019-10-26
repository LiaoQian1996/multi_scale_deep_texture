from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, generator, Synthesis
from lib.ops import *
import math
import time
import numpy as np
import scipy.misc
import cv2
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

parser = argparse.ArgumentParser()

parser.add_argument(
    '--output_dir',
    help = 'output dictionary',
    default = './result/'
)

parser.add_argument(
    '--vgg_ckpt',
    help = 'checkpoint of vgg networks, the check point file of pretrained model should be downloaded',
    default = '/home/liaoqian/DATA/vgg19/vgg_19.ckpt'
)

parser.add_argument(
    '--target_dir',
    help = 'path of target img, texture sample image or style image',
    default = './imgs/tomato.png' 
)

parser.add_argument(
    '--initials',
    help = 'initialized mode of synthesis, come into force only in style_transfer task_mode',
    choices = ['noise', 'content'],
    default = 'noise'
)

parser.add_argument(
    '--top_style_layer',
    help = 'the top layer of vgg network layers used to compute style_loss',
    default = 'VGG54',
    choices = ['VGG11','VGG21','VGG31','VGG41','VGG51','VGG54']
)

parser.add_argument(
    '--texture_shape',
    help = 'img_size of synthesis output texture, if set to [-1,-1], the shape will be \
    the same as sample texture image',
    nargs = '+',
    type = int
)

parser.add_argument(
    '--pyrm_layers',
    help = 'layers number of pyramid',
    default = 6,
    type = int
)

parser.add_argument(
    '--W_tv',
    help = 'weight of total variation loss',
    type = float,
    default = 0.1
)

parser.add_argument(
    '--stddev',
    help = 'standard deviation of noise',
    type = float,
    default = 0.1
)


parser.add_argument(
    '--max_iter',
    help = 'max iteration',
    type = int,
    default = 100,
    required = True
)

parser.add_argument(
    '--print_loss',
    help = 'whether print current loss',
    action = 'store_true'
)

FLAGS = parser.parse_args()
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')
    
# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# pyramid = get_pyramid(targets, pyramid_layers) # a list store the pyramid of target textures image
pyrm_layers = FLAGS.pyrm_layers
start = time.time()
tar_name = FLAGS.target_dir.split('/')[-1]; tar_name = tar_name.split('.')[0]
targets0 = data_loader(FLAGS)

for i in range(pyrm_layers - 1, -1, -1) :
    targets = tf.constant(targets0, dtype = tf.float32)
    w0, h0 = [targets.shape[1], targets.shape[2]]; w, h = [ w0//(2**i), h0//(2**i) ]
    target = tf.image.resize_bicubic(targets, [w, h])
    print('\nCurrent image : ',tar_name)
    print('Image size : ',target.shape)
    print('Now in pyramid layer %i, total %i layer (from L%i to L0)\n'%(i, pyrm_layers, pyrm_layers-1))
    if i == pyrm_layers - 1: # if now initial the large scale gen_output 
        if FLAGS.texture_shape == [-1, -1]:
            FLAGS.texture_shape = [w, h]
        else: 
            FLAGS.texture_shape = [ FLAGS.texture_shape[0]//(2**i), \
                                    FLAGS.texture_shape[1]//(2**i)]
        try:
            gen_output, style_loss = Synthesis(None, target,\
                                   upsampling = False, FLAGS = FLAGS)     
        except:
            raise ValueError('Pyramid is too higher ! ')
    else:
        gen_output, style_loss = Synthesis(gen_output, target,\
                                       upsampling = True, FLAGS = FLAGS) 
    if i == 0:
        total_time = (time.time() - start)
        scipy.misc.toimage(np.squeeze(gen_output), cmin=-1.0, cmax=1.0) \
                        .save(os.path.join(FLAGS.output_dir,'%s_%i_%i_%i_%.1f.png'%
                                           (tar_name, pyrm_layers, \
                                            gen_output.shape[1],gen_output.shape[2],total_time)))
    tf.reset_default_graph()  
        
print('Optimization done !!! ') 