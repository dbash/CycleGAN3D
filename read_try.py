
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import utils
import models
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im
import med_data as md
import matplotlib.pyplot as plt

DEFAULT_TFRECORD_PATH = '/scratch/data/MRCT/TFRecord'
IMG_DEPTH = 30
IMG_SIZE = 256
tfrec_path = DEFAULT_TFRECORD_PATH

a_tfrec_train_file = os.path.join(tfrec_path, 'MR_train.tfrecords')
b_tfrec_train_file = os.path.join(tfrec_path, 'CT_train.tfrecords')
a_tfrec_test_file = os.path.join(tfrec_path, 'MR_test.tfrecords')
b_tfrec_test_file = os.path.join(tfrec_path, 'CT_test.tfrecords')

b_filename = tf.placeholder(tf.string, shape=[None])
b_dataset = tf.data.TFRecordDataset(b_filename).repeat(1)
b_dataset = b_dataset.map(md.parse)
# b_dataset = b_dataset.batch(1)
# b_dataset = b_dataset.shuffle(buffer_size=66)
b_iterator = b_dataset.make_initializable_iterator()
b_real_unsh, b_features = b_iterator.get_next()


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run([b_iterator.initializer], feed_dict={b_filename: [b_tfrec_train_file]})

for i in range(70):
    b_real_val, b_f_val = sess.run([b_real_unsh, b_features])
    print(i, b_real_val.shape, b_real_val.dtype, [b_f_val[x] for x in ['depth', 'height', 'width']])
    # b_real = tf.reshape(b_real_unsh, (1, 30, 100, 100, 1))
    # b_real_val_sh = sess.run(b_real)
    # print(b_real_val_sh.shape, b_real_val_sh.dtype)

sess.close()
