#the code was based on https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch
#modified by Dina Bashkirova (dbash@bu.edu).


from __future__ import absolute_import, division, print_function

import os
import utils
import models
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im
import data_mnist


from glob import glob
import ops



A_DATA_PATH = '/scratch/data/gta/random/sem'
B_DATA_PATH = '/scratch/data/gta/random/rgb'


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', dest='name', default='gta_segm_3d', help='name of the model')
parser.add_argument('--crop_x', dest='crop_x', type=int, default=108, help='then crop to this size')
parser.add_argument('--crop_y', dest='crop_y', type=int, default=192, help='then crop to this size')
parser.add_argument('--channels', dest='channels', type=int, default=3, help='number of channels in a frame (3 for RGB)')

parser.add_argument('--epoch', dest='epoch', type=int, default=800, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--features', dest='features', type=int, default=64, help='number of conv filters in the first layer')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--depth', dest='depth', type=int, default=8, help='number of frames in the sequence')
parser.add_argument('--png', dest='save_png', type=bool, default=False, help='save the results as png images')
parser.add_argument('--cpkt_dir', dest='cpkt_dir', default='/scratch/dinka_model/checkpoints/', help='where to save the checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='/scratch/test_images/', help='where to save the test results')
parser.add_argument('--split', dest='split', default='test', help='train or test')
args = parser.parse_args()

dataset = args.name
crop_x = args.crop_x
crop_y = args.crop_y
channels = args.channels
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
Z_CROP = args.depth
z_offset = (2**(np.ceil(np.log2(Z_CROP))) - Z_CROP) / 2 # generators produce sequences with number of frames equal to a 2^n,
#  so we need to crop some frames if Z_crop  != 2^n
save_png = args.save_png
features = args.features
gpu_id = args.gpu_id
cpkt_dir = args.cpkt_dir
save_dir = args.save_dir
data_type = args.split
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


save_root = save_dir + dataset
a_save_dir = save_root + '/' + data_type + 'A/'
b_save_dir = save_root + '/' + data_type + 'B/'
png_path = os.path.join(save_root, 'png', data_type)
if save_png:
    utils.mkdir(png_path)

utils.mkdir([a_save_dir, b_save_dir])

a_real = tf.placeholder(tf.float32, shape=[1, Z_CROP, crop_x, crop_y, channels])
b_real = tf.placeholder(tf.float32, shape=[1, Z_CROP, crop_x, crop_y, channels])

a2b = models.generator(a_real, 'a2b', channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]
b2a = models.generator(b_real, 'b2a', channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]
b2a2b = models.generator(b2a, 'a2b', reuse=True, channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]
a2b2a = models.generator(a2b, 'b2a', reuse=True, channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = ops.counter()

a_test_img_paths = sorted(glob(A_DATA_PATH + '/' + data_type + '/*.gif'))
b_test_img_paths = sorted(glob(B_DATA_PATH + '/' + data_type + '/*.gif'))
a_test_pool = data_mnist.ImageData(sess, a_test_img_paths, Z_CROP,
                                   crop_size=(crop_x, crop_y), channels=channels, random=False,
                                   return_bname=True)
b_test_pool = data_mnist.ImageData(sess, b_test_img_paths, Z_CROP,
                                   crop_size=(crop_x, crop_y), channels=channels, random=False,
                                   return_bname=True)



saver = tf.train.Saver()
ckpt_path = utils.load_checkpoint(cpkt_dir + dataset, sess, saver)
if ckpt_path is None:
    raise Exception('No checkpoint!')
else:
    print('Copy variables from % s' % ckpt_path)

nit_op = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


for i in range(a_test_pool.img_num): #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    a_real_ipt, bname = a_test_pool.batch_full(depth=30)
    print('bname = ', bname)
    if a_real_ipt.shape[1] % Z_CROP != 0:
        n_pad = Z_CROP - a_real_ipt.shape[1] % Z_CROP
        a_real_ipt_pad = np.concatenate([a_real_ipt,
                                         a_real_ipt[:, -n_pad:-1, ...]],
                                        axis=1)
    else:
        a_real_ipt_pad = a_real_ipt
    a2b_opt_full = np.zeros(a_real_ipt_pad.shape)
    a2b2a_opt_full = np.zeros(a_real_ipt_pad.shape)
    for j in range(a_real_ipt_pad.shape[1]//Z_CROP):
        a_real_part = a_real_ipt_pad[:, j*Z_CROP:(j+1)*Z_CROP, ...]
        [a2b_opt, a2b2a_opt] = sess.run([a2b, a2b2a],
                                        feed_dict={a_real: a_real_part})
        a2b_opt_full[:,j*Z_CROP:(j+1)*Z_CROP,...] = a2b_opt
        a2b2a_opt_full[:, j * Z_CROP:(j + 1) * Z_CROP, ...] = a2b2a_opt

    if save_png:
        im.write_multiple_png([a_real_ipt_pad[0], a2b_opt_full[0], a2b2a_opt_full[0]],
                              png_path, bname,  ['a', 'a2b', 'a2b2a'])
    sample_opt = np.concatenate((a_real_ipt_pad, a2b_opt_full, a2b2a_opt_full), axis=0)

    assert not np.any(np.isnan(sample_opt))

    im.save_array_as_gif(im.immerge3d(sample_opt, 1, 3), '%s/%s.gif' %
                         (a_save_dir,  bname),
                         durn_cf=0.000001)

for i in range(b_test_pool.img_num):
    b_real_ipt, bname = b_test_pool.batch_full(depth=30)
    if b_real_ipt.shape[1] % Z_CROP != 0:
        n_pad = Z_CROP - b_real_ipt.shape[1] % Z_CROP
        b_real_ipt_pad = np.concatenate([b_real_ipt,  b_real_ipt[:, -n_pad:-1, ...]],
                                    axis=1)
    else:
        b_real_ipt_pad = b_real_ipt
    b2a_opt_full = np.zeros(b_real_ipt_pad.shape)
    b2a2b_opt_full = np.zeros(b_real_ipt_pad.shape)
    for j in range(b_real_ipt_pad.shape[1]//Z_CROP):
        b_real_part = b_real_ipt_pad[:, j*Z_CROP:(j+1)*Z_CROP, ...]
        [b2a_opt, b2a2b_opt] = sess.run([b2a, b2a2b],
                                        feed_dict={b_real: b_real_part})
        b2a_opt_full[:,j*Z_CROP:(j+1)*Z_CROP,...] = b2a_opt
        b2a2b_opt_full[:, j * Z_CROP:(j + 1) * Z_CROP, ...] = b2a2b_opt
    if save_png:
        im.write_multiple_png([b_real_ipt_pad[0], b2a_opt_full[0], b2a2b_opt_full[0]],
                              png_path, bname, ['b', 'b2a', 'b2a2b'])
    sample_opt = np.concatenate((b_real_ipt_pad, b2a_opt_full, b2a2b_opt_full), axis=0)


    assert not np.any(np.isnan(sample_opt))

    im.save_array_as_gif(im.immerge3d(sample_opt, 1, 3), '%s/%s.gif' %
                         (b_save_dir, bname),
                         durn_cf=0.000001)








