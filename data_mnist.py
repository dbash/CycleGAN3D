from __future__ import absolute_import, division, print_function

import tensorflow as tf
import imageio as imio
import numpy as np
import os

def read_gif(path, channels=1):
    im = np.array(imio.mimread(path))
    if channels == 3:
        if im.ndim == 3:
            im = np.stack([im, im, im], axis=3)
        else:
            im = im[..., :-1]
    elif channels == 1:
        im = im[..., np.newaxis]
    return im

def read_gif_volume(img_path, depth=30, crop_size=(84, 84), channels=1):
    crop_x, crop_y = crop_size
    img = read_gif(img_path, channels=channels)
    if img.shape[0] > depth:
        rand_idx = np.random.randint(img.shape[0] - depth)
        res_img = img[rand_idx:rand_idx+depth, ...]
    elif img.shape[0] < depth:
        print(img.shape)
        return None
    else:
        res_img = img

    start_x, start_y = (0, 0)
    if res_img.shape[1] > crop_size[0] or res_img.shape[2] > crop_size[1]:
        start_x = np.random.randint(0, (res_img.shape[1] - crop_size[0]) - 1)
        start_y = np.random.randint(0, (res_img.shape[2] - crop_size[1]) - 1)

    res_img = res_img[:, start_x:start_x + crop_size[0], start_y: start_y + crop_size[1], ...]
    res_img = np.reshape(res_img, (1, depth, crop_x, crop_y, channels))

    return res_img


def read_gif_volume_linsp(img_path, depth=30, crop_size=(84, 84), channels=1):
    crop_x, crop_y = crop_size
    img = read_gif(img_path, channels=channels)
    if img.shape[0] > depth:
        mask = np.linspace(start=0, stop=img.shape[0]-1, num=depth)
        #rand_idx = 4 #np.random.randint(img.shape[0] - depth)
        res_img = img[mask]
    elif img.shape[0] < depth:
        print(img.shape)
        return None
    else:
        res_img = img

    start_x, start_y = (0, 0)
    if res_img.shape[1] > crop_size[0] or res_img.shape[2] > crop_size[1]:
        start_x = np.random.randint(0, (res_img.shape[1] - crop_size[0]) - 1)
        start_y = np.random.randint(0, (res_img.shape[2] - crop_size[1]) - 1)

    res_img = res_img[:, start_x:start_x + crop_size[0], start_y: start_y + crop_size[1], ...]
    res_img = np.reshape(res_img, (1, depth, crop_x, crop_y, channels))

    return res_img




class ImageData:

    def __init__(self, session, image_paths, out_depth, crop_size=256, channels=1, random=True,
                 return_bname=False, linsp=False):
        self.sess = session
        self.image_paths = image_paths
        self.depth = out_depth
        self.counter = 0
        self.crop_size = crop_size
        self.channels = channels
        self.return_bname = return_bname
        if random:
            self.shuffle()
        self.img_num = len(image_paths)
        self.img_batch, _ = self.batch()
        self.linsp = linsp

        self.batch_method = read_gif_volume_linsp



    def __len__(self):
        return self.img_num

    def batch_ops(self):
        return self.img_batch

    def batch(self):
        self.counter += 1
        if self.counter == len(self.image_paths):
            self.shuffle()
            self.counter = 0
        #print(len(self.image_paths), self.counter)

        image = read_gif_volume(self.image_paths[self.counter], depth=self.depth,
                                crop_size=self.crop_size, channels=self.channels)
        if image is None:
            self.image_paths.pop(self.counter)
            image = read_gif_volume(self.image_paths[self.counter], depth=self.depth,
                                    crop_size=self.crop_size, channels=self.channels)
            self.img_num -= 1

        image -= image.min()
        image_norm = 2.0*(image/image.max()) - 1.0
        if self.return_bname:
            bname = os.path.basename(self.image_paths[self.counter]).split('.')[0]
            return image_norm, bname
        return image_norm, len(self.image_paths)


    def batch_full(self, depth):
        self.counter += 1
        if self.counter == len(self.image_paths):
            #self.shuffle()
            self.counter = 0
        #print(len(self.image_paths), self.counter)
        image = read_gif_volume_linsp(self.image_paths[self.counter], depth=depth,
                                crop_size=self.crop_size, channels=self.channels)
        if image is None:
            self.image_paths.pop(self.counter)
            image = read_gif_volume_linsp(self.image_paths[self.counter], depth=depth,
                                    crop_size=self.crop_size, channels=self.channels)
            self.img_num -= 1

        image -= image.min()
        image_norm = 2.0*(image/image.max()) - 1.0
        if self.return_bname:
            bname = os.path.basename(self.image_paths[self.counter]).split('.')[0]
            return image_norm, bname
        return image_norm

    def shuffle(self):
        np.random.shuffle(self.image_paths)

