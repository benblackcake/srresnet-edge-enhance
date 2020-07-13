
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
import pywt


def cany_oper(image):
    """Using cany operator to get image edge map"""
    kernel_size = 3
    low_threshold = 1
    high_threshold = 10

    gray_img = rgb2gray(image)
    blur_gray = cv2.GaussianBlur(gray_img,(kernel_size, kernel_size), 0).astype(np.uint8)

    edges = cv2.Canny(blur_gray, 10, 50)

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

def sobel_direct_oper(image):
    ''' 
    Input Y channel Image 
    
    return 3 direction edge map
    '''
    x_k = np.array([1,0,-1,2,0,-2,1,0,-1]).reshape(3,3)
    y_k = np.array([1,2,1,0,0,0,-1,-2,-1]).reshape(3,3)
    diagonal_k = np.array([0,-1,-1,1,0,-1,1,1,0]).reshape(3,3)

    x_edge = cv2.filter2D(image, cv2.CV_32F, x_k, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
    x_edge = cv2.convertScaleAbs(x_edge)

    y_edge = cv2.filter2D(image, cv2.CV_32F, y_k, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
    y_edge = cv2.convertScaleAbs(y_edge)

    diagonal_edge = cv2.filter2D(image, cv2.CV_32F, diagonal_k, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
    diagonal_edge = cv2.convertScaleAbs(diagonal_edge)

    # img = np.expand_dims(img,axis=-1)
    x_edge = np.expand_dims(x_edge,axis=-1)
    y_edge = np.expand_dims(y_edge,axis=-1)
    diagonal_edge = np.expand_dims(diagonal_edge,axis=-1)

    result = np.concatenate([x_edge,y_edge,diagonal_edge], axis=-1) #[:,:,3]

    # print(result.shape)

    return result

def sobel_oper_batch(batch):
    # print(batch.shape) 
    # if batch.shape[1]%2 !=t 0 || batch.shape[2]%2 != 0:
    sobeled = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2]))
    for i in range(batch.shape[0]):
        sobeled[i, :, :] = sobel_oper(batch[i, :, :, :])
    return sobeled

def sobel_direct_oper_batch(batch):
    sobeled = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2],3))
    for i in range(batch.shape[0]):
        sobeled[i,:,:,:] = sobel_direct_oper(batch[i,:,:,:])

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

        batch_hr = batch_bgr2rgb(get_batch)
        dwt_rgb = batch_dwt(batch_hr)
        dwt_rgb = np.clip(np.abs(dwt_rgb), 0, 255).astype('uint8')
        dwt_r_BCD = dwt_rgb[:,:,:,1:4]
        dwt_g_BCD = dwt_rgb[:,:,:,5:8]
        dwt_b_BCD = dwt_rgb[:,:,:,9:12]

        dwt_label = np.concatenate([dwt_r_BCD, dwt_g_BCD, dwt_b_BCD], axis=-1)/255.

        sobeled_batch_r = sobel_direct_oper_batch(np.expand_dims(dwt_rgb[:,:,:,0], axis=-1))
        sobeled_batch_g = sobel_direct_oper_batch(np.expand_dims(dwt_rgb[:,:,:,4], axis=-1))
        sobeled_batch_b = sobel_direct_oper_batch(np.expand_dims(dwt_rgb[:,:,:,8], axis=-1))

        sobeled_train = np.concatenate([sobeled_batch_r,sobeled_batch_g,sobeled_batch_b],axis=-1)/255. # Normalized



        loss += sess.run(loss_function,
                         feed_dict={'srresnet_training:0': False,\
                                    'LR_DWT_edge:0': sobeled_train,\
                                    'HR_DWT_edge:0': dwt_label,\
                                    })
        total += 1
    loss = loss / total
    return loss


def get_data_set(path,label):
    f = h5py.File(path, 'r')
    data = f[label]
    return data


def batch_bgr2ycbcr(batch):

    for i in range(batch.shape[0]):
        batch[i,:,:,:] = cv2.cvtColor(batch[i,:,:,:], cv2.COLOR_BGR2YCR_CB)


    return batch

def batch_bgr2rgb(batch):

    for i in range(batch.shape[0]):
        batch[i,:,:,:] = cv2.cvtColor(batch[i,:,:,:], cv2.COLOR_BGR2RGB)

    return batch

def modcrop(img, scale =2):
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


def batch_dwt(batch):
    dwt_batch = np.zeros([batch.shape[0], batch.shape[1]//2, batch.shape[2]//2, 12])

    for i in range(batch.shape[0]):
        LL_r, (LH_r, HL_r, HH_r) = pywt.dwt2(batch[i,:,:,0], 'haar')
        coeffs_R = np.stack([LL_r,LH_r, HL_r, HH_r],axis=-1)

        LL_g, (LH_g, HL_g, HH_g) = pywt.dwt2(batch[i,:,:,1], 'haar')
        coeffs_G = np.stack([LL_g,LH_g, HL_g, HH_g],axis=-1)

        LL_b, (LH_b, HL_b, HH_b) = pywt.dwt2(batch[i,:,:,2], 'haar')
        coeffs_B = np.stack([LL_b,LH_b, HL_b, HH_b],axis=-1)

        coeffs = np.concatenate([coeffs_R, coeffs_G, coeffs_B], axis=-1)

        dwt_batch[i,:,:,:] = coeffs
        # print(coeffs.shape)
    return dwt_batch

def batch_Idwt(batch):
    '''
    Args:
        batch: Tensor of batch [16,h,w,12]
    Returns:
        Idwt_batch: Tensor of Inverse wavelet transform [16,h*2,w*2,3]
    '''

    dwt_batch = np.zeros([batch.shape[0], batch.shape[1]*2, batch.shape[2]*2, 3])

    for i in range(batch.shape[0]):
        Idwt_R = pywt.idwt2((batch[i,:,:,0],(batch[i,:,:,1],batch[i,:,:,2],batch[i,:,:,3])), wavelet='haar')
        Idwt_G = pywt.idwt2((batch[i,:,:,4],(batch[i,:,:,5],batch[i,:,:,6],batch[i,:,:,7])), wavelet='haar')
        Idwt_B = pywt.idwt2((batch[i,:,:,8],(batch[i,:,:,9],batch[i,:,:,10],batch[i,:,:,11])), wavelet='haar')

        coeffs = cv2.merge([Idwt_R, Idwt_G, Idwt_B])
        dwt_batch[i,:,:,:] = coeffs
        # print(coeffs.shape)
    return dwt_batch

