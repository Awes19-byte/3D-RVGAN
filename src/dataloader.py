import numpy as np
from numpy import load
import tensorflow as tf
from scipy import ndimage

def load_real_data(filename):

    data = load(filename)

    X1, X2 = data['arr_0'], data['arr_1']
    print('aaaaaaaaaaaaaaaaaa',X1.shape)
    print('bbbbbbbbbbbbbbbbbbbb', X2.shape)
    # normalize from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5


    return [X1, X2]

def generate_real_data(data, batch_id, batch_size, patch_shape):

    trainA, trainB = data

    start = batch_id*batch_size
    end = start+batch_size
    X1, X2 = trainA[start:end], trainB[start:end]

    y1 = -np.ones((batch_size, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((batch_size, patch_shape[1], patch_shape[1], 1))
    return [X1, X2], [y1,y2]

def generate_real_data_random(data, random_samples, patch_shape):

    trainA, trainB = data

    id = np.random.randint(0, trainA.shape[0], random_samples)
    X1, X2 = trainA[id], trainB[id]

    y1 = -np.ones((random_samples, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((random_samples, patch_shape[1], patch_shape[1], 1))
    return [X1, X2], [y1,y2]


def generate_fake_data_fine(g_model, batch_data, x_global, patch_shape):

    X = g_model.predict([batch_data,x_global])
    y1 = np.ones((len(X), patch_shape[0],patch_shape[0], patch_shape[0], 1))

    return X, y1

def generate_fake_data_coarse(g_model, batch_data, patch_shape):

    X, X_global = g_model.predict(batch_data)
    y1 = np.ones((len(X), patch_shape[1],patch_shape[1], patch_shape[1], 1))

    return [X,X_global], y1

def resize(X_realA,X_realB,X_realC,out_shape):
    X_realA = tf.image.resize(X_realA, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    X_realA = np.array(X_realA)
    
    X_realB = tf.image.resize(X_realB, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    X_realB = np.array(X_realB)

    X_realC = tf.image.resize(X_realC, out_shape, method=tf.image.ResizeMethod.LANCZOS3)
    X_realC = np.array(X_realC)
    
    return [X_realA,X_realB,X_realC]

def resize_volume(img, x, y, z):

    desired_depth = z
    desired_width = y
    desired_height = x

    current_depth = img.shape[1]
    current_width = img.shape[2]
    current_height = img.shape[3]

    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    imgs = []
    if len(img.shape) > 4:
       for i in img:
          im = ndimage.zoom(i, (depth_factor, width_factor, height_factor, 1), order=1)
          imgs.append(im)
    else:
        for i in img:
            im = ndimage.zoom(i, (depth_factor, width_factor, height_factor), order=1)
            imgs.append(im)


    return np.array(imgs)