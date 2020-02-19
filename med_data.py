from __future__ import absolute_import, division, print_function

import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage.interpolation import rotate
import tensorflow as tf
import matplotlib.pyplot as plt

CROP_SIZE = 100
MARGIN = 50
IMG_SIZE = 256
IMAGE_DEPTH = 30
N_CHANNELS = 1

def read_dicom(dir, normalize=True):
    dclist = sorted([ent.path for ent in os.scandir(dir)])
    dc_img = sitk.GetArrayFromImage(sitk.ReadImage(dclist[0]))
    img_arr = np.zeros((len(dclist), dc_img.shape[1], dc_img.shape[2]))
    for i in range(len(dclist)):
        img_arr[i] = (sitk.GetArrayFromImage(sitk.ReadImage(dclist[i]))[0])
    if normalize:
        img_arr -= img_arr.min()
        img_arr *= 255/img_arr.max()
    return img_arr.astype('int16')

def read_one_dicom(path, normalize=True):
    image = np.array(sitk.GetArrayFromImage(sitk.ReadImage(path))[0])
    if normalize:
        image -= image.min()
        image = (image*255)//image.max()
    return image.astype('int16')

def read_dicoms(dirs, normalize=True):
    dicom_list = map(lambda x: read_dicom(x, normalize), dirs)
    return np.asarray(dicom_list, dtype=float)


def read_dicom_random_pick(dirs, num_images, augment=True, n_augment=5):
    n = len(dirs)
    rand_ind = np.random.shuffle(np.arange(n))[:num_images]
    img_batch = map(lambda x:get_image(x, augment=augment, num_img=n_augment), dirs[rand_ind])
    return np.array(img_batch)


def write_dicom(img_arr, dir):
    for i in range(img_arr.shape[0]):
        dcm_file = os.path.join(dir, str(i + 1).zfill(3) + '.dcm')
        sitk.WriteImage(sitk.GetImageFromArray(img_arr[i]), dcm_file)


def augment_image(img_arr, num_img=5):
    aug_img = np.zeros((num_img, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    aug_img[0] = img_arr
    for i in range(1, num_img):
        ang = np.random.randint(10, 90)
        new_img = rotate(num_img, angle=ang, reshape=False)
        aug_img[i] = new_img
    return aug_img


def get_image(dir, augment=True, num_img=5):
    img_arr = read_dicom(dir)
    if augment:
        return augment_image(img_arr, num_img)
    return img_arr


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_dicoms_to_trfecord(tfrecord_path, data):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for item in data:
        img = read_dicom(item, normalize=True)
        img_str = img.tostring()
        depth, height, width = img.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            'depth': _int64_feature(depth),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channels': _int64_feature(1),
            'image': _bytes_feature(img_str),
        }))

        writer.write(example.SerializeToString())


    writer.close()


def write_dicoms_to_trfecord_2d(tfrecord_path, data):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for item in data:
        img = read_one_dicom(item, normalize=True)
        img_str = img.tostring()
        height, width = img.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channels': _int64_feature(1),
            'image': _bytes_feature(img_str),
        }))

        writer.write(example.SerializeToString())

    writer.close()


def tfrecord_dicom(img_root, train_path, test_path, train_split=0.6):

    img_dir_list = [ent.path for ent in os.scandir(img_root)]
    n_images = len(img_dir_list)
    random_shuffle = np.arange(n_images)
    np.random.shuffle(img_dir_list)
    n_train = int(n_images*train_split)
    train_split = img_dir_list[:n_train]
    test_split = img_dir_list[n_train:-1]

    write_dicoms_to_trfecord(train_path, train_split)
    write_dicoms_to_trfecord(test_path, test_split)



def load_decode_tfrecords(filename_queue, batch_size, img_size=256, img_depth=30):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'depth': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'channels': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image'], tf.int16)
    depth = tf.cast(features['depth'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)

    image_shape = (depth, height, width, channels)
    image = tf.reshape(image, image_shape)
    rot_angle = tf.random_uniform(shape=[1], minval=0, maxval=90, dtype=tf.float32, seed=None, name=None)
    image = tf.contrib.image.rotate(image, np.ones(img_depth)*rot_angle)
    resized_image = tf.image.resize_images(image, (img_size, img_size))
    resized_image = tf.reshape(resized_image[:img_depth], (img_depth, img_size, img_size, N_CHANNELS))
    images = tf.train.shuffle_batch(
        [resized_image], batch_size=batch_size, capacity=30,
        num_threads=2, min_after_dequeue=10
    )
    return images


def parse_orig(example):
    features = tf.parse_single_example(
        example,
        # Defaults are not specified since both keys are required.
        features={
            'depth': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'channels': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image'], tf.int16)
    image = tf.cast(image, dtype=tf.float32)
    image_shape = (IMAGE_DEPTH, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    image = tf.Print(image, [tf.shape(image)], message='IMAGE SHAPE', summarize=100)
    image = tf.reshape(image, image_shape)
    rot_angle = tf.random_uniform(shape=[1], minval=0, maxval=90,
                                  dtype=tf.float32, seed=None, name=None)
    image = tf.contrib.image.rotate(image, np.ones(IMAGE_DEPTH) * rot_angle)
    crop_x = tf.random_uniform(shape=[], minval=0, maxval=IMG_SIZE - CROP_SIZE,
                                   dtype=tf.int32, seed=None, name=None)
    crop_y = tf.random_uniform(shape=[], minval=0, maxval=IMG_SIZE - CROP_SIZE,
                               dtype=tf.int32, seed=None, name=None)
    cropped_img = tf.image.crop_to_bounding_box(
        image, crop_x, crop_y, CROP_SIZE, CROP_SIZE
    )
    cropped_img = tf.div(
        tf.subtract(cropped_img, tf.reduce_min(cropped_img)),
        tf.subtract(tf.reduce_max(cropped_img), tf.reduce_min(cropped_img))
    )

    return cropped_img


def parse(example):
    features = tf.parse_single_example(
        example,
        # Defaults are not specified since both keys are required.
        features={
            'depth': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'channels': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image'], tf.int16)
    image = tf.cast(image, dtype=tf.float32)
    image_shape = (IMAGE_DEPTH, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    image = tf.reshape(image, image_shape)
    # rot_angle = tf.random_uniform(shape=[1], minval=0, maxval=90,
    #                               dtype=tf.float32, seed=None, name=None)
    # image = tf.contrib.image.rotate(image, np.ones(IMAGE_DEPTH) * rot_angle)
    # crop_x = tf.random_uniform(shape=[], minval=0, maxval=IMG_SIZE - CROP_SIZE,
    #                                dtype=tf.int32, seed=None, name=None)
    # crop_y = tf.random_uniform(shape=[], minval=0, maxval=IMG_SIZE - CROP_SIZE,
    #                            dtype=tf.int32, seed=None, name=None)
    # cropped_img = tf.image.crop_to_bounding_box(
    #     image, crop_x, crop_y, CROP_SIZE, CROP_SIZE
    # )
    # cropped_img = tf.div(
    #     tf.subtract(cropped_img, tf.reduce_min(cropped_img)),
    #     tf.subtract(tf.reduce_max(cropped_img), tf.reduce_min(cropped_img))
    # )

    # cropped_img_resh = tf.reshape(cropped_img, (1, 30, CROP_SIZE, CROP_SIZE, 1))
    # cropped_img_resh = print_shape_tf(cropped_img_resh, 'AFTER PARSE')
    return image


tfrecord_dicom('/scratch/data/MRCT/CT/',
            '/scratch/data/MRCT/TFRecords_test/CT_train.tfrecords',
            '/scratch/data/MRCT/TFRecords_test/CT_test.tfrecords', train_split=0.6)


