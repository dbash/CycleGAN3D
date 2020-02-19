#the code was based on https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch
#modified by Dina Bashkirova (dbash@bu.edu).


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ops
import os



import utils
import models
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im
from glob import glob
import data_mnist




""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', dest='name', default='gta_segm_3d', help='name of the model')
parser.add_argument('--A_path', dest='a_path', default='/scratch/data/gta/random/sem', help='path to videos from domain A')
parser.add_argument('--B_path', dest='b_path', default='/scratch/data/gta/random/rgb', help='path to videos from domain B')
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
parser.add_argument('--save_dir', dest='cpkt_dir', default='/scratch/dinka_model/checkpoints/', help='where to save the checkpoints')
parser.add_argument('--train_img_dir', dest='train_img_dir', default='/scratch/sample_images_while_training/', help='where to save sample train images')
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
cpkt_dir = args.cpkt_dir
features = args.features
gpu_id = args.gpu_id
A_DATA_PATH = args.a_path
B_DATA_PATH = args.b_path
save_train_img = args.train_img_dir
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' graph '''
    # nodes
    a_real = tf.placeholder(tf.float32, shape=[batch_size, Z_CROP, crop_x, crop_y, channels])
    b_real = tf.placeholder(tf.float32, shape=[batch_size, Z_CROP, crop_x, crop_y, channels])

    a2b = models.generator(a_real, 'a2b', channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]  # 32 -> 30
    b2a = models.generator(b_real, 'b2a', channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]

    b2a2b = models.generator(b2a, 'a2b', reuse=True, channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]
    a2b2a = models.generator(a2b, 'b2a', reuse=True, channels=channels, gf_dim=features)[:, z_offset:-z_offset, ...]

    check_op = tf.add_check_numerics_ops()

    a2b_sample = tf.placeholder(tf.float32, shape=[batch_size, Z_CROP, crop_x, crop_y, channels])
    b2a_sample = tf.placeholder(tf.float32, shape=[batch_size, Z_CROP, crop_x, crop_y, channels])

    a_dis = models.discriminator(a_real, 'a', df_dim=features)
    b2a_dis = models.discriminator(b2a, 'a', reuse=True, df_dim=features)
    b_dis = models.discriminator(b_real, 'b', df_dim=features)
    a2b_dis = models.discriminator(a2b, 'b', reuse=True, df_dim=features)
    b2a_sample_dis = models.discriminator(b2a_sample, 'a', reuse=True, df_dim=features)
    a2b_sample_dis = models.discriminator(a2b_sample, 'b', reuse=True, df_dim=features)

    # losses

    g_loss_a2b = tf.identity(ops.l2_loss(a2b_dis, tf.ones_like(a2b_dis)), name='g_loss_a2b')
    g_loss_b2a = tf.identity(ops.l2_loss(b2a_dis, tf.ones_like(b2a_dis)), name='g_loss_b2a')

    cyc_loss_a = tf.identity(ops.l1_loss(a_real, a2b2a) * 10.0, name='cyc_loss_a')
    cyc_loss_b = tf.identity(ops.l1_loss(b_real, b2a2b) * 10.0, name='cyc_loss_b')
    g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a + cyc_loss_b

    d_loss_a_real = ops.l2_loss(a_dis, tf.ones_like(a_dis))
    d_loss_b2a_sample = ops.l2_loss(b2a_sample_dis, tf.zeros_like(b2a_sample_dis))
    d_loss_a = tf.identity((d_loss_a_real + d_loss_b2a_sample) / 2.0, name='d_loss_a')
    d_loss_b_real = ops.l2_loss(b_dis, tf.ones_like(b_dis))
    d_loss_a2b_sample = ops.l2_loss(a2b_sample_dis, tf.zeros_like(a2b_sample_dis))
    d_loss_b = tf.identity((d_loss_b_real + d_loss_a2b_sample) / 2.0, name='d_loss_b')

    # summaries
    g_summary = ops.summary_tensors([g_loss_a2b, g_loss_b2a, cyc_loss_a, cyc_loss_b])
    d_summary_a = ops.summary(d_loss_a)
    d_summary_b = ops.summary(d_loss_b)

    ''' optim '''
    t_var = tf.trainable_variables()
    d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
    d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
    g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]

    d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
    d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)



""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = ops.counter()

'''data'''
a_img_paths = glob(A_DATA_PATH + '/train/*.gif')
b_img_paths = glob(B_DATA_PATH + '/train/*.gif')
a_data_pool = data_mnist.ImageData(sess, a_img_paths, Z_CROP,
                                   crop_size=(crop_x, crop_y), channels=channels)
b_data_pool = data_mnist.ImageData(sess, b_img_paths, Z_CROP,
                                   crop_size=(crop_x, crop_y), channels=channels)

a_test_img_paths = glob(A_DATA_PATH + '/test/*.gif')
b_test_img_paths = glob(B_DATA_PATH + '/test/*.gif')
a_test_pool = data_mnist.ImageData(sess, a_test_img_paths, Z_CROP,
                                   crop_size=(crop_x, crop_y), channels=channels)
b_test_pool = data_mnist.ImageData(sess, b_test_img_paths, Z_CROP,
                                   crop_size=(crop_x, crop_y), channels=channels)

'''summary'''
summary_writer = tf.summary.FileWriter('./summaries/' + dataset, sess.graph)

'''saver'''
ckpt_dir = cpkt_dir + dataset
utils.mkdir(ckpt_dir + '/')

saver = tf.train.Saver(max_to_keep=20)
ckpt_path = utils.load_checkpoint(ckpt_dir, sess, saver)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

if ckpt_path is None:
    sess.run(init_op)

else:
    print('Copy variables from % s' % ckpt_path)

'''train'''
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:

    batch_epoch = min(len(a_data_pool), len(b_data_pool))
    max_it = epoch * batch_epoch

    n_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Total number of trainable variables = %d" % n_variables)

    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)
        # prepare data
        a_real_ipt, _ = a_data_pool.batch()
        b_real_ipt, _ = b_data_pool.batch()

        a2b_opt, b2a_opt = sess.run([a2b, b2a],
                                    feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})

        # train G
        g_summary_opt, _ = sess.run([g_summary, g_train_op],
                                    feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        summary_writer.add_summary(g_summary_opt, it)
        # train D_b
        d_summary_b_opt, _ = sess.run([d_summary_b, d_b_train_op],
                                      feed_dict={b_real: b_real_ipt,
                                                 a2b_sample: a2b_opt})
        summary_writer.add_summary(d_summary_b_opt, it)
        # train D_a
        d_summary_a_opt, _ = sess.run([d_summary_a, d_a_train_op],
                                      feed_dict={a_real: a_real_ipt,
                                                 b2a_sample: b2a_opt})
        summary_writer.add_summary(d_summary_a_opt, it)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 2000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' %
                                   (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)


        # sample
        if (it + 1) % 250 == 0:
            a_real_ipt, _ = a_test_pool.batch()
            b_real_ipt, _ = b_test_pool.batch()

            [a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt] = sess.run([a2b, a2b2a, b2a, b2a2b],
                                                feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
            sample_opt = np.concatenate((a_real_ipt, a2b_opt, a2b2a_opt, b_real_ipt,
                                         b2a_opt, b2a2b_opt), axis=0)

            assert not np.any(np.isnan(sample_opt))
            save_dir = save_train_img + dataset
            utils.mkdir(save_dir + '/')
            im.save_array_as_gif(im.immerge3d(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).gif' %
                       (save_dir, epoch, it_epoch, batch_epoch),
                                 durn_cf=0.000001)

except Exception as e:
    coord.request_stop(e)
finally:
    print("Stop threads and close session!")
    coord.request_stop()
    coord.join(threads)
    sess.close()
