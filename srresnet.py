
import tensorflow as tf
from utils import tf_idwt, tf_dwt, tf_batch_ISwt

class Srresnet:
    """Srresnet Model"""

    def __init__(self, training, content_loss='mse', learning_rate=1e-4, num_blocks=32, num_upsamples=2):
        self.learning_rate = learning_rate
        self.num_blocks = num_blocks
        self.num_upsamples = num_upsamples
        self.training = training

        if content_loss not in ['mse', 'L1','edge_loss_mse','edge_loss_L1']:
            print('Invalid content loss function. Must be \'mse\', or \'L1_loss\'.')
            exit()
        self.content_loss = content_loss


    def _Prelu(self, _x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],\
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def ResidualBlock(self, x, kernel_size, filter_size):
        """Residual block a la ResNet"""
        # with tf.variable_scope('sr_edge_net') as scope:		
        weights = {
            'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size,filter_size, filter_size], stddev=1e-3), name='w1_redidual'),
            'w2': tf.Variable(tf.random_normal([kernel_size, kernel_size,filter_size, filter_size], stddev=1e-3), name='w2_residual'),

        }

        skip = x
        x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
        # x = tf.layers.batch_normalization(x, training=self.training)
        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        # x = tf.layers.batch_normalization(x, training=self.training)

        x = x + skip
        return x

    def Upsample2xBlock(self, x, kernel_size,in_channel, filter_size):
        weights = {
            'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size,in_channel, filter_size], stddev=1e-3), name='w1_upsample'),
        }
        """Upsample 2x via SubpixelConv"""
        print('init',x)
        x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
        print('before',x)
        x = tf.depth_to_space(x, 2)
        print('after',x)

        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        return x


    def RDBParams(self):
        weightsR = {}
        biasesR = {}
        D = self.num_blocks
        C = 8
        G = 64
        # G0 = self.G0
        ks = 3

        for i in range(1, D+1):
            for j in range(1, C+1):
                weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=0.01), name='w_R_%d_%d' % (i, j))}) 
                biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
            weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=0.01), name='w_R_%d_%d' % (i, C+1))})
            biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

        return weightsR, biasesR

    def RDBs(self, input_layer):
        rdb_concat = list()
        rdb_in = input_layer

        D = 16
        C = 8
        G = 64
        # G0 = self.G0
        ks = 3
        for i in range(1, D+1):
            x = rdb_in
            for j in range(1, C+1):
                tmp = tf.nn.conv2d(x, self._weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self._biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                x = tf.concat([x, tmp], axis=3)

            x = tf.nn.conv2d(x, self._weightsR['w_R_%d_%d' % (i, C+1)], strides=[1,1,1,1], padding='SAME') +  self._biasesR['b_R_%d_%d' % (i, C+1)]
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)
            edge_branch_out = tf.concat(rdb_concat, axis=3)
            print('__DEBUG__RDBs %s i:%d'%(rdb_in,i))

        return tf.concat(rdb_concat, axis=3) 

    def forward_branch_bine(self, input_LL, input_edge):
        rdb_concat = list()
        rdb_in = input_edge
        x_LL = input_LL

        D = self.num_blocks
        C = 8
        G = 64
        # G0 = self.G0
        ks = 3
        for i in range(1, D+1):
            x_LL = self.ResidualBlock(x_LL, 3, 64)
            x_edge = rdb_in

            for j in range(1, C+1):
                tmp = tf.nn.conv2d(x_edge, self._weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self._biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                x_edge = tf.concat([x_edge, tmp], axis=3)

            x_edge = tf.nn.conv2d(x_edge, self._weightsR['w_R_%d_%d' % (i, C+1)], strides=[1,1,1,1], padding='SAME') +  self._biasesR['b_R_%d_%d' % (i, C+1)]

            rdb_in = tf.add(x_edge, rdb_in)
            rdb_in = tf.add(rdb_in, x_LL)

            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3), x_LL


    def forward(self,x_LL, x_edge):
        with tf.variable_scope('forward_branch',reuse=tf.AUTO_REUSE) as scope:
            weights = {
                'w_resnet_in': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-2), name='w_resnet_in'),
                'w_resnet_1': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-2), name='w_resnet_1'),
                'w_resnet_out': tf.Variable(tf.random_normal([9, 9, 64, 3], stddev=1e-2), name='w_resnet_out'),
                'w_RDB_in': tf.Variable(tf.random_normal([3, 3, 9, 64], stddev=1e-2), name='w_resnet_in'),
                'w_RDB_1': tf.Variable(tf.random_normal([3, 3, self.num_blocks*64, 64], stddev=1e-2), name='w_resnet_in'),
                'w_RDB_out': tf.Variable(tf.random_normal([3, 3, 64, 9], stddev=1e-2), name='w_resnet_in'),

            }
            biases = {
                'b1': tf.Variable(tf.zeros([64], name='b1')),
                'b2': tf.Variable(tf.zeros([64], name='b2')),
                'b3': tf.Variable(tf.zeros([9], name='b3')),
                }

            self._weightsR, self._biasesR = self.RDBParams()

            x_edge = tf.nn.conv2d(x_edge, weights['w_RDB_in'], strides=[1,1,1,1], padding='SAME') + biases['b1']
            x_edge_skip = x_edge

            x_LL = tf.nn.conv2d(x_LL, weights['w_resnet_in'], strides=[1,1,1,1], padding='SAME')
            x_LL = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x_LL)
            x_LL_skip = x_LL

            x_edge, x_LL = self.forward_branch_bine(x_edge, x_LL)
            print('-----------=====debug',x_edge)

            x_edge = tf.nn.conv2d(x_edge, weights['w_RDB_1'], strides=[1,1,1,1], padding='SAME') + biases['b2']
            print('-----------=====debug',x_edge)

            x_edge =  tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x_edge)
            x_edge = tf.add(x_edge_skip, x_edge)

            x_LL = tf.nn.conv2d(x_LL, weights['w_resnet_1'], strides=[1,1,1,1], padding='SAME', name='layer_1')
            # x_LL = tf.layers.batch_normalization(x_LL, training=self.training)
            x_LL = tf.add(x_LL_skip, x_LL)


            for i in range(self.num_upsamples):
                x_edge = self.Upsample2xBlock(x_edge, kernel_size=3, in_channel=64, filter_size=256)
                x_LL = self.Upsample2xBlock(x_LL, kernel_size=3, in_channel=64, filter_size=256)

            x_edge = tf.nn.conv2d(x_edge, weights['w_RDB_out'], strides=[1,1,1,1], padding='SAME') + biases['b3']
            x_LL = tf.nn.conv2d(x_LL, weights['w_resnet_out'], strides=[1,1,1,1], padding='SAME', name='y_predict')

            tf_swt_debug_RA = tf.expand_dims(x_LL[:,:,:,0], axis=-1)
            tf_swt_debug_GA = tf.expand_dims(x_LL[:,:,:,1], axis=-1)
            tf_swt_debug_BA = tf.expand_dims(x_LL[:,:,:,2], axis=-1)

            y_RA_pred = tf.concat([tf_swt_debug_RA,x_edge[:,:,:,0:3]], axis=-1)
            y_GA_pred = tf.concat([tf_swt_debug_GA,x_edge[:,:,:,3:6]], axis=-1)
            y_BA_pred = tf.concat([tf_swt_debug_BA,x_edge[:,:,:,6:9]], axis=-1)

            y_pred = tf.concat([y_RA_pred, y_GA_pred, y_BA_pred], axis=-1)

            y_pred = tf_batch_ISwt(y_pred)

            return x_LL, x_edge, y_pred
                
    def _content_loss(self, y_A, y_A_pred, y_BCD, y_BCD_pred, hr, sr_pred):

        # tf_dwt_debug = tf_dwt(y_A_pred)
        # tf_dwt_debug_RA = tf.expand_dims(y_A_pred[:,:,:,0], axis=-1)
        # tf_dwt_debug_GA = tf.expand_dims(y_A_pred[:,:,:,1], axis=-1)
        # tf_dwt_debug_BA = tf.expand_dims(y_A_pred[:,:,:,2], axis=-1)

        # print('__DEBUG__tf_dwt_debug', tf_dwt_debug_RA)

        # y_RA_pred = tf.concat([tf_dwt_debug_RA,y_BCD_pred[:,:,:,0:3]], axis=-1)
        # y_GA_pred = tf.concat([tf_dwt_debug_GA,y_BCD_pred[:,:,:,3:6]], axis=-1)
        # y_BA_pred = tf.concat([tf_dwt_debug_BA,y_BCD_pred[:,:,:,6:9]], axis=-1)

        # y_idwt_pred = tf_idwt(tf.concat([y_RA_pred, y_GA_pred, y_BA_pred], axis=-1))
        """MSE, VGG22, or VGG54"""
        if self.content_loss == 'mse':
            return tf.reduce_mean(tf.square(y_A - y_A_pred))

        if self.content_loss == 'L1':
            return tf.reduce_mean(tf.abs(y_A - y_A_pred))

        if self.content_loss == 'edge_loss_mse':
            lamd = 0.05
            # y_sobeled = tf.image.sobel_edges(y)
            # y_pred_sobeled = tf.image.sobel_edges(y_pred)
            return tf.reduce_mean(tf.square(y_A - y_A_pred)) + (lamd*tf.reduce_mean(tf.square(y_BCD - y_BCD_pred)))

        if self.content_loss == 'edge_loss_L1':
            lamd = 0.05
            # y_sobeled = tf.image.sobel_edges(y)
            # y_pred_sobeled = tf.image.sobel_edges(y_pred)
            return tf.cast(tf.reduce_mean(tf.square(hr - sr_pred)), tf.float32) + 5e-1* (tf.reduce_mean(tf.square(y_A - y_A_pred))) + (2e-1*tf.reduce_mean(tf.abs(y_BCD - y_BCD_pred)))
        # if self.content_loss == 'edge_loss_L1':
        #     lamd = 2.0
        #     # y_sobeled = tf.image.sobel_edges(y)
        #     # y_pred_sobeled = tf.image.sobel_edges(y_pred)
        #     return tf.reduce_mean(tf.square(y_A - y_A_pred)) + (lamd*tf.reduce_mean(tf.abs(y_BCD - y_BCD_pred)))

    def loss_function(self, y_A, y_A_pred, y_BCD, y_BCD_pred, hr, sr_pred):

        # Content loss only
        return self._content_loss(y_A, y_A_pred, y_BCD, y_BCD_pred, hr, sr_pred)

    def optimize(self, loss):
        # tf.control_dependencies([discrim_train
        # update_ops needs to be here for batch normalization to work
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='forward_branch')
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward_branch'))


class SRGanDiscriminator:
    """
    SRGAN Discriminator Model from Ledig et. al. 2017

    """

    def __init__(self, training, learning_rate=1e-4, image_size=96):
        self.graph_created = False
        self.learning_rate = learning_rate
        self.training = training
        self.image_size = image_size

    def ConvolutionBlock(self, x, kernel_size, filters, strides):
        """Conv2D + BN + LeakyReLU"""
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same',
                             use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x

    def forward(self, x):
        """Builds the forward pass network graph"""
        with tf.variable_scope('discriminator') as scope:
            # Reuse variables when graph is applied again
            if self.graph_created:
                scope.reuse_variables()
            self.graph_created = True

            # Image dimensions are fixed to the training size because of the FC layer
            x.set_shape([None, self.image_size, self.image_size, 3])

            x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same')
            x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)

            x = self.ConvolutionBlock(x, 3, 64, 2)
            x = self.ConvolutionBlock(x, 3, 128, 1)
            x = self.ConvolutionBlock(x, 3, 128, 2)
            x = self.ConvolutionBlock(x, 3, 256, 1)
            x = self.ConvolutionBlock(x, 3, 256, 2)
            x = self.ConvolutionBlock(x, 3, 512, 1)
            x = self.ConvolutionBlock(x, 3, 512, 2)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(x)
            logits = tf.layers.dense(x, 1)
            x = tf.sigmoid(logits)
            return x, logits

    def loss_function(self, y_real_pred, y_fake_pred, y_real_pred_logits, y_fake_pred_logits):
        """Discriminator wants to maximize log(y_real) + log(1-y_fake)."""
        loss_real = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real_pred_logits), y_real_pred_logits))
        loss_fake = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake_pred_logits), y_fake_pred_logits))
        return loss_real + loss_fake

    def optimize(self, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
