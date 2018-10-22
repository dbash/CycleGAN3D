from __future__ import absolute_import, division, print_function

import tensorflow as tf
from PIL import Image, ImageSequence
import numpy as np

def read_gif_volume(img_path, depth=30, crop_size=84, channels=1):
    img = Image.open(img_path)
    frames = [np.asarray(frame.copy()) for frame in ImageSequence.Iterator(img)]
    img_arr = np.array(frames)
    n = len(frames)

    if n > depth:
        idx_include = np.linspace(0, n - 1, depth, dtype=int)
        res_img = img_arr[idx_include, ...]

    else:
        res_img = img_arr

    res_img = np.reshape(res_img, (1, depth, crop_size, crop_size, channels))
    return res_img




class ImageData:

    def __init__(self, session, image_paths, out_depth, crop_size=256, channels=1):
        self.sess = session
        self.image_paths = image_paths
        self.depth = out_depth
        self.counter = 0
        self.crop_size = crop_size
        self.channels = channels

        self.img_batch, self.img_num = self.batch()


    def __len__(self):
        return self.img_num

    def batch_ops(self):
        return self.img_batch

    def batch(self):
        self.counter += 1
        if self.counter == len(self.image_paths):
            self.shuffle()
            self.counter = 0
        image = read_gif_volume(self.image_paths[self.counter], depth=self.depth,
                                crop_size=self.crop_size, channels=self.channels)
        image -= image.min()
        image_norm = 2.0*(image/image.max()) - 1.0
        return image_norm, len(self.image_paths)

    def shuffle(self):
        np.random.shuffle(self.image_paths)

