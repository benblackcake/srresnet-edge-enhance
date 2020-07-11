
import tensorflow as tf
import argparse
from benchmark import Benchmark
import os
import sys
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import srresnet
from utils import (downsample_batch,build_log_dir,preprocess,evaluate_model,batch_bgr2ycbcr,
                   get_data_set, sobel_oper_batch, cany_oper_batch, sobel_direct_oper_batch,
                   tf_dwt)
import numpy as np
import pywt
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
    parser.add_argument('--load-gen', type=str, help='Checkpoint to load generator weights only from.')
    parser.add_argument('--name', type=str, help='Name of experiment.')
    parser.add_argument('--overfit', action='store_true', help='Overfit to a single image.')
    parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size.')
    parser.add_argument('--log-freq', type=int, default=10000,
                        help='How many training iterations between validation/checkpoints.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--content-loss', type=str, default='mse', choices=['mse', 'L1','edge_loss_mse','edge_loss_L1'],
                        help='Metric to use for content loss.')
    parser.add_argument('--use-gan', action='store_true',
                        help='Add adversarial loss term to generator and trains discriminator.')
    parser.add_argument('--image-size', type=int, default=96, help='Size of random crops used for training samples.')
    parser.add_argument('--vgg-weights', type=str, default='vgg_19.ckpt',
                        help='File containing VGG19 weights (tf.slim)')
    parser.add_argument('--train-dir', type=str, help='Directory containing training images')
    parser.add_argument('--validate-benchmarks', action='store_true',
                        help='If set, validates that the benchmarking metrics are correct for the images provided by the authors of the SRGAN paper.')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    parser.add_argument('--epoch', type=int, default='1000000', help='How many iterations ')
    parser.add_argument('--is-val', action='store_true', help='How many iterations ')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    srresnet_training = tf.placeholder(tf.bool, name='srresnet_training')

    srresnet_model = srresnet.Srresnet(training=srresnet_training,\
                              learning_rate=args.learning_rate,\
                              content_loss=args.content_loss)

    lr_dwt_edge = tf.placeholder(tf.float32, [None, None, None, 9], name='LR_DWT_edge')
    hr_dwt_edge = tf.placeholder(tf.float32, [None, None, None, 9], name='HR_DWT_edge')


    sr_pred = srresnet_model.forward(lr_dwt_edge)
    sr_loss = srresnet_model.loss_function(hr_dwt_edge, sr_pred)
    sr_opt = srresnet_model.optimize(sr_loss)

    benchmarks = [
        Benchmark('Benchmarks/Set5', name='Set5'),
        Benchmark('Benchmarks/Set14', name='Set14'),
        Benchmark('Benchmarks/BSD100', name='BSD100')
    ]

    if args.validate_benchmarks:
        for benchmark in benchmarks:
            benchmark.validate()

    # Create log folder
    if args.load and not args.name:
        log_path = os.path.dirname(args.load)
    else:
        log_path = build_log_dir(args, sys.argv)

    train_data_path = 'done_dataset\PreprocessedData.h5'
    val_data_path = 'done_dataset\PreprocessedData_val.h5'
    eval_data_path = 'done_dataset\PreprocessedData_eval.h5'


    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        iteration = 0
        epoch = 0

        saver = tf.train.Saver(max_to_keep=100)

        # Load all
        if args.load:
            iteration = int(args.load.split('-')[-1])
            saver.restore(sess, args.load)
            print(saver)
            print("load_process_DEBUG")


        train_data_set = get_data_set(train_data_path,'train')
        val_data_set = get_data_set(val_data_path,'val')
        eval_data_set = get_data_set(eval_data_path,'eval')

        val_error_li =[]
        eval_error_li =[]
        fig = plt.figure()
        
        if args.is_val:
            benchmarks = [
                Benchmark('Benchmarks/Set5', name='Set5'),
                Benchmark('Benchmarks/Set14', name='Set14'),
                Benchmark('Benchmarks/BSD100', name='BSD100'),
                Benchmark('Benchmarks/UCMerced_LandUse', name='UCMerced_LandUse'),
                Benchmark('Benchmarks/RSSCN7', name='RSSCN7')
            ]

            log_line = ''
            for benchmark in benchmarks:
                psnr, ssim, _, _ = benchmark.evaluate(sess, sr_pred, log_path, iteration)
                print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
                log_line += ',%.7f, %.7f' % (psnr, ssim)
            print()
            # Write to log
            with open(log_path + '/PSNR.csv', 'a') as f:
                f.write(
                    'iteration, set5_psnr, set5_ssim, set14_psnr, set14_ssim, bsd100_psnr, bsd100_ssim,UCMerced_LandUse_psnr, UCMerced_LandUse_ssim,RSSCN7_psnr, RSSCN7_ssim\n'
                 )
                f.write('%d,%s\n' % (iteration, log_line))
            # Save checkpoint
            # saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)

        else:
            while True:
                t =trange(0, len(train_data_set) - args.batch_size + 1, args.batch_size, desc='Iterations')
                #One epoch 
                for batch_idx in t:
                    t.set_description("Training... [Iterations: %s]" % iteration)
                    
                    # Each 10000 times evaluate model
                    if iteration % args.log_freq == 0:
                        # Loop over eval dataset
                        # for batch_idx in range(0, len(val_data_set) - args.batch_size + 1, args.batch_size): 
                        # # Test every log-freq iterations
                        #     val_error = evaluate_model(sr_loss, val_data_set[batch_idx:batch_idx + 16], sess, 119, args.batch_size)
                        #     eval_error = evaluate_model(sr_loss, eval_data_set[batch_idx:batch_idx + 16], sess, 119, args.batch_size)
                        # val_error_li.append(val_error)
                        # eval_error_li.append(eval_error)

                        # # Log error
                        # plt.plot(val_error_li)
                        # plt.savefig('val_error.png')
                        # plt.plot(eval_error_li)
                        # plt.savefig('eval_error.png')
                        # # fig.savefig()

                        # print('[%d] Test: %.7f, Train: %.7f' % (iteration, val_error, eval_error), end='')
                        # Evaluate benchmarks
                        log_line = ''
                        for benchmark in benchmarks:
                            psnr, ssim, _, _ = benchmark.evaluate(sess, sr_pred, log_path, iteration)
                        #     # benchmark.evaluate(sess, sr_pred, log_path, iteration)
                            print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
                        #     log_line += ',%.7f, %.7f' % (psnr, ssim)
                        # print()
                        # # Write to log
                        with open(log_path + '/loss.csv', 'a') as f:
                            f.write('%d, %s\n' % (iteration, log_line))
                        # Save checkpoint
                        saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)
                    
                    # Train SRResnet   
                    batch_hr = train_data_set[batch_idx:batch_idx + 16]
                    ycbcr_batch = batch_bgr2ycbcr(batch_hr)

                    batch_hr_y = np.expand_dims(ycbcr_batch[:,:,:,0], axis=-1) #Get batch Y channel image
                    batch_hr_cr = np.expand_dims(ycbcr_batch[:,:,:,1], axis=-1) #Get batch cr channel image
                    batch_hr_cb = np.expand_dims(ycbcr_batch[:,:,:,2], axis=-1) #Get batch cb channel image

                    # for i in range(batch_hr.shape[0]):
                    #     cv2.imshow('__DEBUG__',batch_hr[i,:,:,:])
                    #     cv2.waitKey(0)

                    dwt_y_channel = tf_dwt(np.float32(batch_hr_y/255.), in_size=[16,96,96,1])
                    dwt_cr_channel = tf_dwt(np.float32(batch_hr_cr/255.), in_size=[16,96,96,1])
                    dwt_cb_channel = tf_dwt(np.float32(batch_hr_cb/255.), in_size=[16,96,96,1])
                    
                    A_y_prime = tf.expand_dims(dwt_y_channel[:,:,:,0], axis=-1).eval()*255.
                    B_y_prime = tf.expand_dims(dwt_y_channel[:,:,:,1], axis=-1)
                    C_y_prime = tf.expand_dims(dwt_y_channel[:,:,:,2], axis=-1)
                    D_y_prime = tf.expand_dims(dwt_y_channel[:,:,:,3], axis=-1)

                    A_cr_prime = tf.expand_dims(dwt_cr_channel[:,:,:,0], axis=-1).eval()*255.
                    B_cr_prime = tf.expand_dims(dwt_cr_channel[:,:,:,1], axis=-1)
                    C_cr_prime = tf.expand_dims(dwt_cr_channel[:,:,:,2], axis=-1)
                    D_cr_prime = tf.expand_dims(dwt_cr_channel[:,:,:,3], axis=-1)

                    A_cb_prime = tf.expand_dims(dwt_cb_channel[:,:,:,0], axis=-1).eval()*255.
                    B_cb_prime = tf.expand_dims(dwt_cb_channel[:,:,:,1], axis=-1)
                    C_cb_prime = tf.expand_dims(dwt_cb_channel[:,:,:,2], axis=-1)
                    D_cb_prime = tf.expand_dims(dwt_cb_channel[:,:,:,3], axis=-1)
                    # A = tf.cast(tf.clip_by_value(tf.abs(A),0,255), dtype=tf.uint8)
                    A_y_prime = np.clip(np.abs(A_y_prime),0,255).astype(np.uint8)
                    A_cr_prime = np.clip(np.abs(A_cr_prime),0,255).astype(np.uint8)
                    A_cb_prime = np.clip(np.abs(A_cb_prime),0,255).astype(np.uint8)

                    tf.concat
                    concat_y_BCD = tf.concat([B_y_prime,C_y_prime,D_y_prime], axis=-1)
                    concat_cr_BCD = tf.concat([B_cr_prime,C_cr_prime,D_cr_prime], axis=-1)
                    concat_cb_BCD = tf.concat([B_cb_prime,C_cb_prime,D_cb_prime], axis=-1)

                    concat_dwt_hr = tf.concat([concat_y_BCD, concat_cr_BCD, concat_cb_BCD], axis=-1)
                    # print('__DEBBUG__A shape: ',A_prime.shape)
                    # print(concat_BCD)


                    sobeled_batch_y_lr = sobel_direct_oper_batch(A_y_prime)/255.
                    sobeled_batch_cr_lr = sobel_direct_oper_batch(A_cr_prime)/255.
                    sobeled_batch_cb_lr = sobel_direct_oper_batch(A_cb_prime)/255.

                    concat_sobel = np.concatenate([sobeled_batch_y_lr, sobeled_batch_cr_lr, sobeled_batch_cb_lr], axis=-1)

                    # print(concat_dwt_hr.shape)
                    # print(concat_sobel.shape)


                    _, err = sess.run([sr_opt,sr_loss],\
                         feed_dict={srresnet_training: True,\
                                    lr_dwt_edge: concat_sobel,\
                                    hr_dwt_edge: concat_dwt_hr.eval(),\

                                    })

                    #print('__training__ %s' % iteration)
                    iteration += 1
                print('__epoch__: %s' % epoch)
                epoch += 1

if __name__ == "__main__":
    main()
