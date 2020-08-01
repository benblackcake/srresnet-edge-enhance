
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
    # x_edge = np.expand_dims(x_edge,axis=-1)
    # y_edge = np.expand_dims(y_edge,axis=-1)
    # diagonal_edge = np.expand_dims(diagonal_edge,axis=-1)

    result = np.stack([image,x_edge,y_edge,diagonal_edge], axis=-1) #[:,:,3]

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
    sobeled = np.zeros((batch.shape[0], batch.shape[1] , batch.shape[2],4))
    for i in range(batch.shape[0]):
        sobeled[i,:,:,:] = sobel_direct_oper(batch[i,:,:])

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
    return scipy.misc.imresize(image, 1.0 / float(factor), interp='bicubic')


def downsample_batch(batch, factor):
    downsampled = np.zeros((batch.shape[0], batch.shape[1] // factor, batch.shape[2] // factor, 3))
    for i in range(batch.shape[0]):
        downsampled[i, :, :, :] = downsample(batch[i, :, :, :], factor)
    return downsampled


def up_sample(image, factor):
    """Downsampling function which matches photoshop"""
    return scipy.misc.imresize(image, float(factor), interp='bicubic')


def up_sample_batch(batch, factor):
    upsampled = np.zeros((batch.shape[0], batch.shape[1] * factor, batch.shape[2] * factor, 3))
    for i in range(batch.shape[0]):
        # print(batch[i, :, :, :].shape)
        upsampled[i, :, :, :] = up_sample(batch[i, :, :, :], factor)
    # print(upsampled.shape)
    return upsampled

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
        batch_dwt_lr = batch_dwt(batch_hr)


        batch_dwt_A = np.stack([batch_dwt_lr[:,:,:,0], batch_dwt_lr[:,:,:,4], batch_dwt_lr[:,:,:,8]], axis=-1)

        # batch_dwt_A[:,:,:,0] /= np.abs(batch_dwt_A[:,:,:,0]).max()
        # batch_dwt_A[:,:,:,1] /= np.abs(batch_dwt_A[:,:,:,1]).max()
        # batch_dwt_A[:,:,:,2] /= np.abs(batch_dwt_A[:,:,:,2]).max()

        # batch_dwt_A[:,:,:,0] *= 255.
        # batch_dwt_A[:,:,:,1] *= 255.
        # batch_dwt_A[:,:,:,2] *= 255.

        batch_dwt_lr_A = batch_dwt(batch_dwt_A)

        batch_hr_BCD = np.concatenate([batch_dwt_lr[:,:,:,1:4], batch_dwt_lr[:,:,:,5:8], batch_dwt_lr[:,:,:,9:12]], axis=-1)
        batch_lr_BCD = np.concatenate([up_sample_batch(batch_dwt_lr_A[:,:,:,1:4], factor=2), up_sample_batch(batch_dwt_lr_A[:,:,:,5:8], factor=2), up_sample_batch(batch_dwt_lr_A[:,:,:,9:12], factor=2)], axis=-1)
        # batch_lr = downsample_batch(batch_hr, factor=4)
        # batch_lr_BCD = up_sample_batch(batch_lr_BCD, factor=2)

        batch_hr_BCD = batch_hr_BCD/255.
        batch_lr_BCD = batch_lr_BCD/255.
        # print('__DEBUG__hr_BCD',batch_hr_BCD.shape)
        # print('__DEBUG__lr_BCD',batch_lr_BCD.shape)

        # batch_hr_A = np.stack([batch_dwt_hr[:,:,:,0], batch_dwt_hr[:,:,:,4], batch_dwt_hr[:,:,:,8]], axis=-1)
        # batch_lr_A = np.stack([batch_dwt_lr[:,:,:,0], batch_dwt_lr[:,:,:,4], batch_dwt_lr[:,:,:,8]], axis=-1)
        # # print(batch_dwt_hr[:,:,:,1:4].shape)
        # batch_hr_BCD = np.concatenate([batch_dwt_hr[:,:,:,1:4], batch_dwt_hr[:,:,:,5:8], batch_dwt_hr[:,:,:,9:12]], axis=-1)
        # batch_lr_BCD = np.concatenate([batch_dwt_lr[:,:,:,1:4], batch_dwt_lr[:,:,:,5:8], batch_dwt_lr[:,:,:,9:12]], axis=-1)
        # print(batch_hr_BCD.shape)

        # print('debug shape')
        # print(batch_hr_A.shape)
        # print(batch_lr_A.shape)
        # print(batch_hr_BCD.shape)
        # print(batch_lr_BCD.shape)

        # batch_hr = batch_bgr2rgb(get_batch)
        # dwt_rgb = batch_dwt(batch_hr)
        # dwt_rgb = np.clip(np.abs(dwt_rgb), 0, 255).astype('uint8')
        # dwt_r_BCD = dwt_rgb[:,:,:,1:4]
        # dwt_g_BCD = dwt_rgb[:,:,:,5:8]
        # dwt_b_BCD = dwt_rgb[:,:,:,9:12]

        # dwt_label = np.concatenate([dwt_r_BCD, dwt_g_BCD, dwt_b_BCD], axis=-1)/255.
        # dwt_label = dwt_rgb

        # sobeled_batch_r = sobel_direct_oper_batch(dwt_rgb[:,:,:,0])
        # sobeled_batch_g = sobel_direct_oper_batch(dwt_rgb[:,:,:,4])
        # sobeled_batch_b = sobel_direct_oper_batch(dwt_rgb[:,:,:,8])

        # sobeled_batch_r = np.concatenate([sobeled_batch_r,np.expand_dims(dwt_rgb[:,:,:,0], axis=-1)],axis=-1)
        # sobeled_batch_g = np.concatenate([sobeled_batch_g,np.expand_dims(dwt_rgb[:,:,:,4], axis=-1)],axis=-1)
        # sobeled_batch_b = np.concatenate([sobeled_batch_b,np.expand_dims(dwt_rgb[:,:,:,8], axis=-1)],axis=-1)

        # sobeled_train = np.concatenate([sobeled_batch_r,sobeled_batch_g,sobeled_batch_b],axis=-1)/255. # Normalized



        loss += sess.run(loss_function,
                         feed_dict={'srresnet_training:0': False,\
                                    'LR_DWT_A:0': batch_dwt_A,\
                                    'LR_DWT_edge:0': batch_lr_BCD,\
                                    # 'HR_DWT_A:0': batch_hr_A,\
                                    'HR_DWT_edge:0': batch_hr_BCD,\
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


def batch_dwt(batch):
    '''
    Args:
        batch: Input batch RGB image [batch_size, img_h, img_w, 3]
    Returns:
        dwt_batch: Batch  DWT result [batch_size, img_h, img_w, 12]
    '''
    # print(len(batch.shape))
    assert (len(batch.shape) == 4 ),"Input batch Shape error"
    assert (batch.shape[3] == 3 ),"Color channel error"

    dwt_batch = np.zeros([batch.shape[0], batch.shape[1]//2, batch.shape[2]//2, 12])

    for i in range(batch.shape[0]):
        LL_r, (LH_r, HL_r, HH_r) = pywt.dwt2(batch[i,:,:,0], 'haar')
        coeffs_R = np.stack([LL_r,LH_r, HL_r, HH_r], axis=-1)

        LL_g, (LH_g, HL_g, HH_g) = pywt.dwt2(batch[i,:,:,1], 'haar')
        coeffs_G = np.stack([LL_g,LH_g, HL_g, HH_g], axis=-1)

        LL_b, (LH_b, HL_b, HH_b) = pywt.dwt2(batch[i,:,:,2], 'haar')
        coeffs_B = np.stack([LL_b,LH_b, HL_b, HH_b], axis=-1)

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


def dwt_shape(img):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        # print(h)
        # print(w)
        h = (h//2+(h//2%8))*2
        w = (w//2+(w//2%8))*2

        # print(w)
        # print(w-img.shape[1])
        # img = np.pad(img ,pad_width=((h-img.shape[0], w-img.shape[1],0)), 'constant', constant_values=0)
        img = np.pad(array=img, pad_width=(((h-img.shape[0])//2,(h-img.shape[0])//2),((w-img.shape[1])//2, (w-img.shape[1])//2),(0,0)), mode='constant', constant_values=(0,0))
        # img = img[0:h, 0:w, :]
        # print('__DEBUG__')
        # print(h)
        # print(w)
        # print(img.shape)
        # cv2.imshow('t', img)
        # cv2.waitKey(0)
    else:
        h, w = img.shape
        h = (h//2+(h//2%8))*2
        w = (w//2+(w//2%8))*2
        img = img[0:h, 0:w]
    return img



def tf_dwt(yl,  wave='haar'):
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

    filts = filts[None, :, :, :, :]

    filter = tf.convert_to_tensor(filts)
    sz = 2 * (len(w.dec_lo) // 2 - 1)

    with tf.variable_scope('DWT'):

        ### Pad odd length images
        # if in_size[0] % 2 == 1 and tf.shape(yl)[1] % 2 == 1:
        #     yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz + 1], [sz, sz + 1], [0, 0]]), mode='reflect')
        # elif in_size[0] % 2 == 1:
        #     yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz + 1], [sz, sz], [0, 0]]), mode='reflect')
        # elif in_size[1] % 2 == 1:
        #     yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz], [sz, sz + 1], [0, 0]]), mode='reflect')
        # else:
        yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz], [sz, sz], [0, 0]]), mode='reflect')

        y = tf.expand_dims(yl, 1)
        inputs = tf.split(y, [1]*int(y.shape.dims[4]), 4)
        inputs = tf.concat([x for x in inputs], 1)

        outputs_3d = tf.nn.conv3d(inputs, filter, padding='VALID', strides=[1, 1, 2, 2, 1])
        outputs = tf.split(outputs_3d, [1] * int(outputs_3d.shape.dims[1]), 1)
        outputs = tf.concat([x for x in outputs], 4)

        outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2],
                                       tf.shape(outputs)[3], tf.shape(outputs)[4]))

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
    filts = filts[None, :, :, :, :]
    filter = tf.convert_to_tensor(filts)
    s = 2 * (len(w.dec_lo) // 2 - 1)
    out_size = tf.shape(y)[1]

    with tf.variable_scope('IWT'):
        y = tf.expand_dims(y, 1)
        inputs = tf.split(y, [4] * int(int(y.shape.dims[4])/4), 4)
        inputs = tf.concat([x for x in inputs], 1)

        outputs_3d = tf.nn.conv3d_transpose(inputs, filter, output_shape=[tf.shape(y)[0], tf.shape(inputs)[1],
                                                                          2*(out_size-1)+np.shape(ll)[0],
                                                                          2*(out_size-1)+np.shape(ll)[0], 1],
                                            padding='VALID', strides=[1, 1, 2, 2, 1])
        outputs = tf.split(outputs_3d, [1] * int(int(y.shape.dims[4])/4), 1)
        outputs = tf.concat([x for x in outputs], 4)

        outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2],
                                       tf.shape(outputs)[3], tf.shape(outputs)[4]))
        outputs = outputs[:, s: 2 * (out_size - 1) + np.shape(ll)[0] - s, s: 2 * (out_size - 1) + np.shape(ll)[0] - s,
                  :]
    return outputs


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    img_arr1 = np.array(im1).astype('float32')
    img_arr2 = np.array(im2).astype('float32')
    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    psnr = tf.constant(255**2, dtype=tf.float32)/mse
    result = tf.constant(10, dtype=tf.float32)*log10(psnr)
    with tf.Session():
        result = result.eval()
    return result

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))