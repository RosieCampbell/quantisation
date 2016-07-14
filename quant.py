import os
import sys
import time
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
NUM_CHANNELS = 3
PIXEL_DEPTH = 255

def crop_and_resize(np_array):
    imgs = []
    for i, im in enumerate(np_array):
        im = np.swapaxes(im,0,1)
        lx, ly, lz = im.shape
        if(lx > ly):
            im = scipy.misc.imresize(im, (int((IMAGE_SIZE/float(ly))*lx),IMAGE_SIZE))
        if(ly > lx):
            im = scipy.misc.imresize(im, (IMAGE_SIZE,int((IMAGE_SIZE/float(lx))*ly)))
        lx, ly, lz = im.shape
        im = im[lx/2-IMAGE_SIZE/2:lx/2+IMAGE_SIZE/2, ly/2-IMAGE_SIZE/2:ly/2+IMAGE_SIZE/2]
        im = np.swapaxes(im,0,1)
        imgs.append(im)
    return imgs

def read_binary(filename):
    data = np.load(filename + '.npy')
    return data

def write_binary(np_array, filename):
    np.save(filename, np_array)

def read_images(folder):
    images = []
    for f in os.listdir(folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(scipy.misc.imread(folder+"/"+f))
    return images

def write_image(np_array, filename):
    for i, im in enumerate(np_array):
        scipy.misc.imsave(filename + str(i) + '.png', im)
    return

def quantize_images():
    imgs = read_binary('small_original')
    print(imgs[0].shape)
    for i, image in enumerate(imgs):
        for w in range(image.shape[0]):
            for h in range(image.shape[1]):
                 for c in range(image.shape[2]):
                     image[w,h,c] = image[w,h,c] >> 4
                     image[w,h,c] = image[w,h,c] << 4
                     imgs[i] = image
    write_image(imgs, 'small_banded/out')
    write_binary(image, 'small_banded')


def generateDataset():
    data = read_images('large_original')
    data = crop_and_resize(data)
    write_binary(data, "small_original")
    write_image(data, 'small_original/img')
    quantize_images()

generateDataset()
