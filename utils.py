
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
        batch_hr = get_batch
        # batch_hr = np.expand_dims(batch_hr[:,:,:,0], axis=-1) #Get batch Y channel image 
        ycbcr_batch = batch_bgr2ycbcr(batch_hr)
        
        batch_hr_y = np.expand_dims(ycbcr_batch[:,:,:,0], axis=-1) #Get batch Y channel image
        batch_hr_cr = np.expand_dims(ycbcr_batch[:,:,:,1], axis=-1) #Get batch cr channel image
        batch_hr_cb = np.expand_dims(ycbcr_batch[:,:,:,2], axis=-1) #Get batch cb channel image

        dwt_y_channel = tf_dwt(np.float32(batch_hr_y/255.), in_size=[16,96,96,1])
        dwt_cr_channel = tf_dwt(np.float32(batch_hr_cr/255.), in_size=[16,96,96,1])
        dwt_cb_channel = tf_dwt(np.float32(batch_hr_cb/255.), in_size=[16,96,96,1])
        
        A_y_prime = sess.run(tf.expand_dims(dwt_y_channel[:,:,:,0], axis=-1))*255.
        B_y_prime = sess.run(tf.expand_dims(dwt_y_channel[:,:,:,1], axis=-1))
        C_y_prime = sess.run(tf.expand_dims(dwt_y_channel[:,:,:,2], axis=-1))
        D_y_prime = sess.run(tf.expand_dims(dwt_y_channel[:,:,:,3], axis=-1))

        A_cr_prime = sess.run(tf.expand_dims(dwt_cr_channel[:,:,:,0], axis=-1))*255.
        B_cr_prime = sess.run(tf.expand_dims(dwt_cr_channel[:,:,:,1], axis=-1))
        C_cr_prime = sess.run(tf.expand_dims(dwt_cr_channel[:,:,:,2], axis=-1))
        D_cr_prime = sess.run(tf.expand_dims(dwt_cr_channel[:,:,:,3], axis=-1))

        A_cb_prime = sess.run(tf.expand_dims(dwt_cb_channel[:,:,:,0], axis=-1))*255.
        B_cb_prime = sess.run(tf.expand_dims(dwt_cb_channel[:,:,:,1], axis=-1))
        C_cb_prime = sess.run(tf.expand_dims(dwt_cb_channel[:,:,:,2], axis=-1))
        D_cb_prime = sess.run(tf.expand_dims(dwt_cb_channel[:,:,:,3], axis=-1))
        # A = tf.cast(tf.clip_by_value(tf.abs(A),0,255), dtype=tf.uint8)
        A_y_prime = np.clip(np.abs(A_y_prime),0,255).astype(np.uint8)
        A_cr_prime = np.clip(np.abs(A_cr_prime),0,255).astype(np.uint8)
        A_cb_prime = np.clip(np.abs(A_cb_prime),0,255).astype(np.uint8)


        concat_y_BCD = np.concatenate([B_y_prime,C_y_prime,D_y_prime], axis=-1)
        concat_cr_BCD = np.concatenate([B_cr_prime,C_cr_prime,D_cr_prime], axis=-1)
        concat_cb_BCD = np.concatenate([B_cb_prime,C_cb_prime,D_cb_prime], axis=-1)

        concat_dwt_hr = np.concatenate([concat_y_BCD, concat_cr_BCD, concat_cb_BCD], axis=-1)
        # print('__DEBBUG__A shape: ',A_prime.shape)
        # print(concat_BCD)


        sobeled_batch_y_lr = sobel_direct_oper_batch(A_y_prime)/255.
        sobeled_batch_cr_lr = sobel_direct_oper_batch(A_cr_prime)/255.
        sobeled_batch_cb_lr = sobel_direct_oper_batch(A_cb_prime)/255.

        concat_sobel = np.concatenate([sobeled_batch_y_lr, sobeled_batch_cr_lr, sobeled_batch_cb_lr], axis=-1)


        loss += sess.run(loss_function,
                         feed_dict={'srresnet_training:0': False,\
                                    'LR_DWT_edge:0': concat_sobel,\
                                    'HR_DWT_edge:0': concat_dwt_hr,\
                                    })
        total += 1
    loss = loss / total
    return loss


def get_data_set(path,label):
    f = h5py.File(path, 'r')
    data = f[label]
    return data


def batch_bgr2ycbcr(batch):
    # output = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2], batch.shape[3]))
    # output[:,:,:,0] = 16+((batch[:,:,:,2]*65.481)/255+(batch[:,:,:,1]*128.553)/255+(batch[:,:,:,0]*24.966)/255)
    # output[:,:,:,1] = 128-((batch[:,:,:,2]*-39.97)/255-(batch[:,:,:,1]*74.203)/255+(batch[:,:,:,0]*112.0)/255)
    # output[:,:,:,2] = 128+((batch[:,:,:,2]*112.0)/255-(batch[:,:,:,1]*93.786)/255-(batch[:,:,:,0]*24.966)/255)

    for i in range(batch.shape[0]):
        batch[i,:,:,:] = cv2.cvtColor(batch[i,:,:,:], cv2.COLOR_BGR2YCrCb)
        # output[i,:,:,:] = tmp
        # print(output[i,:,:,:])
        # cv2.imshow('testing...', batch[i,:,:,:])
        # cv2.waitKey(0)

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

def tf_dwt(yl,  in_size, wave='haar'):
    w = pywt.Wavelet(wave)
    ll = np.outer(w.dec_lo, w.dec_lo)
    lh = np.outer(w.dec_hi, w.dec_lo)
    hl = np.outer(w.dec_lo, w.dec_hi)
    hh = np.outer(w.dec_hi, w.dec_hi)
    d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    d_temp[::-1, ::-1, 0, 0] = ll
    d_temp[::-1, ::-1, 0, 1] = lh
    d_temp[::-1, ::-1, 0, 2] = hl
    d_temp[::-1, ::-1, 0, 3] = hh

    filts = d_temp.astype('float32')
    filts = np.copy(filts)
    filter = tf.convert_to_tensor(filts)
    sz = 2 * (len(w.dec_lo) // 2 - 1)

    with tf.variable_scope('DWT'):

        # Pad odd length images
        if in_size[0] % 2 == 1 and tf.shape(yl)[1] % 2 == 1:
            yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz + 1], [sz, sz + 1], [0, 0]]), mode='reflect')
        elif in_size[0] % 2 == 1:
            yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz + 1], [sz, sz], [0, 0]]), mode='reflect')
        elif in_size[1] % 2 == 1:
            yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz], [sz, sz + 1], [0, 0]]), mode='reflect')
        else:
            yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz], [sz, sz], [0, 0]]), mode='reflect')

        # group convolution
        outputs = tf.nn.conv2d(yl[:, :, :, 0:1], filter, padding='VALID', strides=[1, 2, 2, 1])
        for channel in range(1, int(yl.shape.dims[3])):
            temp = tf.nn.conv2d(yl[:, :, :, channel:channel+1], filter, padding='VALID', strides=[1, 2, 2, 1])
            outputs = tf.concat([outputs, temp], axis=3)

    return outputs


def tf_idwt(y,  wave='haar'):
    w = pywt.Wavelet(wave)
    ll = np.outer(w.rec_lo, w.rec_lo)
    lh = np.outer(w.rec_hi, w.rec_lo)
    hl = np.outer(w.rec_lo, w.rec_hi)
    hh = np.outer(w.rec_hi, w.rec_hi)
    d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    d_temp[:, :, 0, 0] = ll
    d_temp[:, :, 0, 1] = lh
    d_temp[:, :, 0, 2] = hl
    d_temp[:, :, 0, 3] = hh
    filts = d_temp.astype('float32')
    filter = tf.convert_to_tensor(filts)
    s = 2 * (len(w.dec_lo) // 2 - 1)

    with tf.variable_scope('IWT'):
        out_size = tf.shape(y)[1]
        in_t = tf.slice(y, (0, 0, 0, 0),
                           (tf.shape(y)[0], out_size, out_size, 4))

        outputs = tf.nn.conv2d_transpose(in_t, filter, output_shape=[tf.shape(y)[0], 2*(out_size-1)+np.shape(ll)[0],
                                                                     2*(tf.shape(y)[1]-1)+np.shape(ll)[0], 1],
                                         padding='VALID', strides=[1, 2, 2, 1])
        for channels in range(4, int(y.shape.dims[-1]), 4):
            y_batch = tf.slice(y, (0, 0, 0, channels), (tf.shape(y)[0], out_size, out_size, 4))
            out_t = tf.nn.conv2d_transpose(y_batch, filter, output_shape=[tf.shape(y)[0], 2*(out_size-1)+np.shape(ll)[0],
                                                                     2*(out_size-1)+np.shape(ll)[0], 1],
                                           padding='VALID', strides=[1, 2, 2, 1])
            outputs = tf.concat((outputs, out_t), axis=3)
        outputs = outputs[:, s: 2*(out_size-1)+np.shape(ll)[0]-s, s: 2*(out_size-1)+np.shape(ll)[0]-s, :]
    return outputs