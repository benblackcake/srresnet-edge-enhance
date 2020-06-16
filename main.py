
import tensorflow as tf
import argparse
from benchmark import Benchmark
import os
import sys
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
import srresnet
from utils import downsample_batch,build_log_dir,preprocess,evaluate_model,get_data_set, sobel_oper_batch
import numpy as np

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

    hr_y = tf.placeholder(tf.float32, [None, None, None, 3], name='HR_image')
    hr_edge = tf.placeholder(tf.float32, [None, None, None, 1], name='HR_edge')
    lr_edge = tf.placeholder(tf.float32, [None, None, None, 1], name='LR_edge')
    lr_x = tf.placeholder(tf.float32, [None, None, None, 3], name='LR_image')

    sr_pred,sr_edge_pred = srresnet_model.forward(lr_x, lr_edge)
    sr_loss = srresnet_model.loss_function(hr_y, sr_pred, hr_edge, sr_edge_pred)
    sr_opt = srresnet_model.optimize(sr_loss)

    benchmarks = [Benchmark('Benchmarks/Set5', name='Set5'),
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

        saver = tf.train.Saver()

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
                Benchmark('Benchmarks/UCMerced_LandUse_S', name='UCMerced_LandUse_S')
            ]

            log_line = ''
            for benchmark in benchmarks:
                psnr, ssim, _, _ = benchmark.evaluate(sess, sr_pred, log_path, iteration)
                print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
                log_line += ',%.7f, %.7f' % (psnr, ssim)
            print()
            # Write to log
            # with open(log_path + '/loss.csv', 'a') as f:
                # f.write('%d,\n' % (iteration, log_line))
            # Save checkpoint
            # saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)

        else:
            while True:
                t =trange(0, len(train_data_set) - args.batch_size + 1, args.batch_size, desc='Iterations')
                #One epoch 
                for batch_idx in t:
                    t.set_description("Training... [Iterations: %s]" % iteration)
                    
                    #Each 10000 times evaluate model
                    if iteration % args.log_freq == 0:
                        #Loop over eval dataset
                        for batch_idx in range(0, len(val_data_set) - args.batch_size + 1, args.batch_size): 
                        # Test every log-freq iterations
                            val_error = evaluate_model(sr_loss, val_data_set[batch_idx:batch_idx + 16], sess, 119, args.batch_size)
                            eval_error = evaluate_model(sr_loss, eval_data_set[batch_idx:batch_idx + 16], sess, 119, args.batch_size)
                        val_error_li.append(val_error)
                        eval_error_li.append(eval_error)

                        # Log error
                        plt.plot(val_error_li)
                        plt.savefig('val_error.png')
                        plt.plot(eval_error_li)
                        plt.savefig('eval_error.png')
                        # fig.savefig()

                        print('[%d] Test: %.7f, Train: %.7f' % (iteration, val_error, eval_error), end='')
                        # Evaluate benchmarks
                        log_line = ''
                        for benchmark in benchmarks:
                            psnr, ssim, _, _ = benchmark.evaluate(sess, sr_pred, log_path, iteration)
                            print(' [%s] PSNR: %.2f, SSIM: %.4f' % (benchmark.name, psnr, ssim), end='')
                            log_line += ',%.7f, %.7f' % (psnr, ssim)
                        print()
                        # Write to log
                        with open(log_path + '/loss.csv', 'a') as f:
                            f.write('%d, %.15f, %.15f%s\n' % (iteration, val_error, eval_error, log_line))
                        # Save checkpoint
                        saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)
                    
                    # Train Srresnet   
                    batch_hr = train_data_set[batch_idx:batch_idx + 16]
                    batch_hr_edge = sobel_oper_batch(batch_hr)
                    batch_hr_edge = np.expand_dims(batch_hr_edge,axis=-1)/255. #normalize

                    batch_lr = downsample_batch(batch_hr, factor=4)
                    batch_lr_edge = sobel_oper_batch(batch_lr)
                    batch_lr_edge = np.expand_dims(batch_lr_edge,axis=-1)/255. #normalize

                    batch_lr, batch_hr = preprocess(batch_lr, batch_hr)

                    _, err = sess.run([sr_opt,sr_loss],\
                         feed_dict={srresnet_training: True,\
                                    lr_x: batch_lr,\
                                    hr_y: batch_hr,\
                                    lr_edge: batch_lr_edge,\
                                    hr_edge: batch_hr_edge
                                    })

                    #print('__training__ %s' % iteration)
                    iteration += 1
                print('__epoch__: %s' % epoch)
                epoch += 1

if __name__ == "__main__":
    main()
