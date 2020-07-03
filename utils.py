
import tensorflow as tf
import numpy as np
import pickle
import skimage.transform
import skimage.filters
import datetime
import os
import shutil
import math
from scipy import misc
import scipy.ndimage
import glob
import h5py
import cv2
from skimage.color import rgb2gray


def cany_oper(image):
    """Using cany operator to get image edge map"""
    kernel_size = 3
    low_threshold = 1
    high_threshold = 10

    gray_img = rgb2gray(image)
    blur_gray = cv2.GaussianBlur(gray_img,(kernel_size, kernel_size), 0).astype(np.uint8)

    edges = cv2.Canny(blur_gray, 30, 150)

    return edges

def sobel_oper(image):
    image = rgb2gray(image)

    x = cv2.Sobel(image,cv2.CV_16S,1,0)
    y = cv2.Sobel(image,cv2.CV_16S,0,1)
     
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
     
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0).astype(np.uint8)
    # print("__sobeled shape__")
    # print(dst.shape)
    return dst 

def sobel_oper_batch(batch):
    # print(batch.shape) 
    # if batch.shape[1]%2 !=t 0 || batch.shape[2]%2 != 0:
    sobeled = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2]))
    for i in range(batch.shape[0]):
        sobeled[i, :, :] = sobel_oper(batch[i, :, :, :])
    return sobeled

def cany_oper_batch(batch):
    # print(batch.shape) 
    # if batch.shape[1]%2 !=t 0 || batch.shape[2]%2 != 0:
    canyed = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2]))
    for i in range(batch.shape[0]):
        canyed[i, :, :] = cany_oper(batch[i, :, :, :])
    return canyed

def downsample(image, factor):
    """Downsampling function which matches photoshop"""
    return scipy.misc.imresize(image, 1.0 / factor, interp='bicubic')


def downsample_batch(batch, factor):
    downsampled = np.zeros((batch.shape[0], batch.shape[1] // factor, batch.shape[2] // factor, 3))
    for i in range(batch.shape[0]):
        downsampled[i, :, :, :] = downsample(batch[i, :, :, :], factor)
    return downsampled

def build_log_dir(args, arguments):
    """Set up a timestamped directory for results and logs for this training session"""
    if args.name:
        log_path = args.name  # (name + '_') if name else ''
    else:
        log_path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join('results', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('Logging results for this session in folder "%s".' % log_path)
    # Output csv header
    with open(log_path + '/loss.csv', 'a') as f:
        f.write(
            'iteration, val_error, eval_error, set5_psnr, set5_ssim, set14_psnr, set14_ssim, bsd100_psnr, bsd100_ssim\n')
    # Copy this code to folder
    shutil.copy2('srresnet.py', os.path.join(log_path, 'srresnet.py'))
    shutil.copy2('main.py', os.path.join(log_path, 'main.py'))
    shutil.copy2('utils.py', os.path.join(log_path, 'utils.py'))
    # Write command line arguments to file
    with open(log_path + '/args.txt', 'w+') as f:
        f.write(' '.join(arguments))
    return log_path


def preprocess(lr, hr):
    """Preprocess lr and hr batch"""
    lr = lr / 255.0
    hr = (hr / 255.0) * 2.0 - 1.0
    return lr, hr


def evaluate_model(loss_function, get_batch, sess, num_images, batch_size):
    """Tests the model over all num_images using input tensor get_batch"""
    loss = 0
    total = 0
    for i in range(int(math.ceil(num_images / batch_size))):
        batch_hr = get_batch
        batch_hr_edge = cany_oper_batch(batch_hr)
        batch_hr_edge = np.expand_dims(batch_hr_edge,axis=-1)/255. #normalize

        batch_lr = downsample_batch(batch_hr, factor=4)

        batch_lr_edge = cany_oper_batch(batch_lr)
        batch_lr_edge = np.expand_dims(batch_lr_edge,axis=-1)/255. #normalize

        batch_lr, batch_hr = preprocess(batch_lr, batch_hr)
        loss += sess.run(loss_function,
                         feed_dict={'srresnet_training:0': False,\
                                    'LR_image:0': batch_lr,\
                                    'HR_image:0': batch_hr,\
                                    'LR_edge:0' : batch_lr_edge,\
                                    'HR_edge:0' : batch_hr_edge
                                    })
        total += 1
    loss = loss / total
    return loss


def get_data_set(path,label):
    f = h5py.File(path, 'r')
    data = f[label]
    return data


def modcrop(img, scale =4):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is gray

    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w]
    return img