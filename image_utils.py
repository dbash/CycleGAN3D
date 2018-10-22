"""
Some codes are modified from https://github.com/Newmu/dcgan_code

These functions are all based on [-1.0, 1.0] image
"""

from __future__ import absolute_import, division, print_function
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os, utils


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """
    transform images from [-1.0, 1.0] to [min_value, max_value] of dtype
    """
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def im2uint(images):
    """ transform images from [-1.0, 1.0] to uint8 """
    return to_range(images, 0, 255, np.uint8)


def im2float(images):
    """ transform images from [-1.0, 1.0] to [0.0, 1.0] """
    return to_range(images, 0.0, 1.0)


def float2uint(images):
    """ transform images from [0, 1.0] to uint8 """
    assert \
        np.min(images) >= 0.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [0.0, 1.0]!'
    return (images * 255).astype(np.uint8)


def uint2float(images):
    """ transform images from uint8 to [0.0, 1.0] of float64 """
    assert images.dtype == np.uint8, 'The input image type should be uint8!'
    return images / 255.0


def imread(path, mode='RGB'):
    """
    read an image into [-1.0, 1.0] of float64

    `mode` can be one of the following strings:

    * 'L' (8 - bit pixels, black and white)
    * 'P' (8 - bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8 - bit pixels, true color)
    * 'RGBA' (4x8 - bit pixels, true color with transparency mask)
    * 'CMYK' (4x8 - bit pixels, color separation)
    * 'YCbCr' (3x8 - bit pixels, color video format)
    * 'I' (32 - bit signed integer pixels)
    * 'F' (32 - bit floating point pixels)
    """
    return scipy.misc.imread(path, mode=mode) / 127.5 - 1


def read_images(path_list, mode='RGB'):
    """ read a list of images into [-1.0, 1.0] and return the numpy array batch in shape of N * H * W (* C) """
    images = [imread(path, mode) for path in path_list]
    return np.array(images)


def imwrite(image, path):
    """ save an [-1.0, 1.0] image """
    image -= image.min()
    image *= 255/image.max()
    return scipy.misc.imsave(path, image)


def imshow(image, slice=1):
    """ show a [-1.0, 1.0] image """
    plt.imshow(image[slice, ...], cmap=plt.gray())
    plt.show()

def save_array_as_gif(img_arr,gifpath, durn_cf=0.08):
    n_slices = img_arr.shape[0]
    img_arr -= img_arr.min()
    img_arr *= 255/img_arr.max()
    img_arr_int = img_arr.astype('uint8')
    durn = n_slices*durn_cf
    with imageio.get_writer(gifpath, mode='I', duration=durn) as writer:
        for i in range(n_slices):
            writer.append_data(img_arr_int[i])
    print("Saved image to %s."%gifpath)

def write_multiple_png(images, root, num, prefix):
    if not os.path.exists(root):
        os.mkdir(root)
    for i in range(len(images)):
        fld = os.path.join(root, prefix[i])
        if not os.path.exists(fld):
            os.mkdir(fld)
        fld_idx = os.path.join(fld, num)
        if not os.path.exists(fld_idx):
            os.mkdir(fld_idx)

        volume = images[i]
        volume -= volume.min()
        volume *= 255/volume.max()
        volume = np.asarray(volume, dtype='uint8')
        for j in range(volume.shape[0]):
            flname = os.path.join(fld_idx, str(j).zfill(2) + '.png')
            imageio.imwrite(flname, volume[j])
            #print("Saved image %s" % flname)


def rgb2gray(images):
    if images.ndim == 4 or images.ndim == 3:
        assert images.shape[-1] == 3, 'Channel size should be 3!'
    else:
        raise Exception('Wrong dimensions!')

    return (images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114).astype(images.dtype)


def imresize(image, size, interp='bilinear'):
    """
    Resize an [-1.0, 1.0] image.

    size : int, float or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : str, optional
        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic'
        or 'cubic').
    """

    # scipy.misc.imresize should deal with uint8 image, or it would cause some problem (scale the image to [0, 255])
    return (scipy.misc.imresize(im2uint(image), size, interp=interp) / 127.5 - 1).astype(image.dtype)



def immerge3d(images, row, col):

    if images.ndim == 5:
        c = images.shape[4]
        d, h, w = images.shape[1], images.shape[2], images.shape[3]
        if c > 1:
            img = np.zeros((d, h * row, w * col, c))
        else:
            img = np.zeros((d, h * row, w * col))
        for idx, image in enumerate(images):
            i = idx % col
            j = idx // col
            if c==1:
                image = image[..., 0]
            img[:, j * h:j * h + h, i * w:i * w + w] = image
    return img


def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * Z * H * W(* C=1 or 3)
    """
    if images.ndim == 4:
        c = images.shape[3]
    elif images.ndim == 3:
        c = 1

    h, w = images.shape[1], images.shape[2]
    if c > 1:
        img = np.zeros((h * row, w * col, c))
    else:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):

        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img


def center_crop(x, crop_z, crop_h=None, crop_w=None):
    if crop_w is None:
        crop_w = crop_z
        crop_h = crop_z
    z, h, w = x.shape[:3]
    k = int(round((z - crop_z) / 2.))
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return x[k: k + crop_z, j:j + crop_h, i:i + crop_w]


def transform(image, resize_z=None, resize_h=None, resize_w=None, interp='bilinear', crop_z=None, crop_h=None, crop_w=None, crop_fn=center_crop):
    """
    transform a [-1.0, 1.0] image with crop and resize

    'crop_fn' should be in the form of crop_fn(x, crop_h, crop_w=None)
    """
    if crop_h is not None:
        image = crop_fn(image, crop_z, crop_h, crop_w)

    if resize_h is not None:
        if resize_w is None:
            resize_w = resize_h
        imresize(image, [resize_z, resize_h, resize_w], interp=interp)

    return image


def imread_transform(path, mode='RGB', resize_z=None,resize_h=None, resize_w=None, interp='bilinear',
                     crop_z=None, crop_h=None, crop_w=None, crop_fn=center_crop):
    """
    read and transform an image into [-1.0, 1.0] of float64

    `mode` can be one of the following strings:

    * 'L' (8 - bit pixels, black and white)
    * 'P' (8 - bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8 - bit pixels, true color)
    * 'RGBA' (4x8 - bit pixels, true color with transparency mask)
    * 'CMYK' (4x8 - bit pixels, color separation)
    * 'YCbCr' (3x8 - bit pixels, color video format)
    * 'I' (32 - bit signed integer pixels)
    * 'F' (32 - bit floating point pixels)

    'crop_fn' should be in the form of crop_fn(x, crop_h, crop_w=None)
    """
    return transform(imread(path, mode), resize_z, resize_h, resize_w, interp, crop_z, crop_h, crop_w, crop_fn)


def read_transform_images(path_list, mode='RGB', resize_z=None, resize_h=None, resize_w=None, interp='bilinear',
                          crop_h=None, crop_w=None, crop_fn=center_crop):
    """ read and transform a list images into [-1.0, 1.0] of float64 and return the numpy array batch in shape of N * H * W (* C) """
    images = [imread_transform(path, mode, resize_z, resize_h, resize_w, interp, crop_h, crop_w, crop_fn) for path in path_list]
    return np.array(images)


if __name__ == '__main__':
    pass
