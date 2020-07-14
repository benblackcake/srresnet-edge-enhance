
import tensorflow as tf


class Srresnet:
    """Srresnet Model"""

    def __init__(self, training, content_loss='mse', learning_rate=1e-4, num_blocks=16, num_upsamples=1):
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
        x = tf.layers.batch_normalization(x, training=self.training)
        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.layers.batch_normalization(x, training=self.training)

        x = x + skip
        return x

    def Upsample2xBlock(self, x, kernel_size, filter_size):
        weights = {
            'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size,64, filter_size], stddev=1e-3), name='w1_upsample'),
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
        D = 3
        C = 2
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

        D = 3
        C = 2
        G = 64
        # G0 = self.G0
        ks = 3
        for i in range(1, D+1):
            x = rdb_in
            print(x)
            for j in range(1, C+1):
                tmp = tf.nn.conv2d(x, self._weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self._biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                x = tf.concat([x, tmp], axis=3)

            x = tf.nn.conv2d(x, self._weightsR['w_R_%d_%d' % (i, C+1)], strides=[1,1,1,1], padding='SAME') +  self._biasesR['b_R_%d_%d' % (i, C+1)]
            rdb_in = tf.add(x, rdb_in)
            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3) 


    def forward(self, x):
        '''
        Args:
            x: Input tensor include 3 direction sobel edge [batch_size, img_h, img_w, 3]
        Returns:
            x_conv_out: SRResnet result but not have UpSample blocks
        '''
        with tf.variable_scope('srresnet_edge',reuse=tf.AUTO_REUSE) as scope:
            # x = tf.concat([x, x_edge],axis=3, name='x_input_concate')
            input_x = x
            weights = {
                'w_in': tf.Variable(tf.random_normal([9, 9, 12, 64], stddev=1e-3), name='w_in'),
                'w1': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-3), name='w1'),
                'w_out': tf.Variable(tf.random_normal([9, 9, 64, 12], stddev=1e-3), name='w_out'),
            }


            x = tf.nn.conv2d(x, weights['w_in'], strides=[1,1,1,1], padding='SAME')
            # x = tf.nn.relu(x)
            x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
            skip = x

            for i in range(self.num_blocks):
                x = self.ResidualBlock(x, 3, 64)

            x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME', name='layer_1')
            x = tf.layers.batch_normalization(x, training=self.training)
            x = x + skip

            for i in range(self.num_upsamples):
                x = self.Upsample2xBlock(x, kernel_size=3, filter_size=256)

            x_conv_out = tf.nn.conv2d(x, weights['w_out'], strides=[1,1,1,1], padding='SAME', name='y_predict')

            # x_conv_out = x_conv_out + input_x
            return x_conv_out

    def _content_loss(self, y, y_pred):
        """MSE, VGG22, or VGG54"""
        if self.content_loss == 'mse':
            return tf.reduce_mean(tf.square(y - y_pred))

        if self.content_loss == 'L1':
            return tf.reduce_mean(tf.abs(y - y_pred))

        # if self.content_loss == 'edge_loss_mse':
        #     lamd = 0.5
        #     # y_sobeled = tf.image.sobel_edges(y)
        #     # y_pred_sobeled = tf.image.sobel_edges(y_pred)
        #     return tf.reduce_mean(tf.square(y - y_pred)) + (lamd*tf.reduce_mean(tf.square(y_edge - y_edge_pred)))

        # if self.content_loss == 'edge_loss_L1':
        #     lamd = 0.5
        #     # y_sobeled = tf.image.sobel_edges(y)
        #     # y_pred_sobeled = tf.image.sobel_edges(y_pred)
        #     return tf.reduce_mean(tf.abs(y - y_pred)) + (lamd*tf.reduce_mean(tf.square(y_edge - y_edge_pred)))

    def loss_function(self, y, y_pred):

        # Content loss only
        return self._content_loss(y, y_pred)

    def optimize(self, loss):
        # tf.control_dependencies([discrim_train
        # update_ops needs to be here for batch normalization to work
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='srresnet_edge')
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='srresnet_edge'))