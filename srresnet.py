
import tensorflow as tf


class Srresnet:
    """Srresnet Model"""

    def __init__(self, training, content_loss='mse', learning_rate=1e-4, num_blocks=16, num_upsamples=2):
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
            'w1':tf.get_variable(name='w1_redidual',\
                shape=[kernel_size, kernel_size, filter_size, filter_size], dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer()),
            'w2':tf.get_variable(name='w2_residual',\
                shape=[kernel_size, kernel_size, filter_size, filter_size], dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer()),
        }

        skip = x
        x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
        x = tf.layers.batch_normalization(x, training=self.training)
        x = self._Prelu(x)
        x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
        x = self._Prelu(x)
        x = tf.layers.batch_normalization(x, training=self.training)

        x = x + skip
        return x

    def Upsample2xBlock(self, x, kernel_size, filter_size):
        weights = {
        'w1':tf.get_variable(name='w1_upsample',\
            shape=[kernel_size, kernel_size, 64, filter_size], dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer()),
        }
        """Upsample 2x via SubpixelConv"""
        print('init',x)
        x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
        print('before',x)
        x = tf.depth_to_space(x, 2)
        print('after',x)

        x = self._Prelu(x)
        return x


    def forward(self, x):
        with tf.variable_scope('srresnet_edge',reuse=tf.AUTO_REUSE) as scope:

            weights = {
                'w_in':tf.get_variable(name='w_in', shape=[9, 9, 3, 64], dtype=tf.float32,\
                    initializer=tf.contrib.layers.xavier_initializer()),
                'w1':tf.get_variable(name='w1', shape=[3, 3, 64, 64], dtype=tf.float32,\
                    initializer=tf.contrib.layers.xavier_initializer()),
                'w_out':tf.get_variable(name='w_out', shape=[9, 9, 64, 3], dtype=tf.float32,\
                    initializer=tf.contrib.layers.xavier_initializer()),
            }

            # print(x_concate)
            x = tf.nn.conv2d(x, weights['w_in'], strides=[1,1,1,1], padding='SAME')
            x = self._Prelu(x)
            skip = x

            for i in range(self.num_blocks):
                x = self.ResidualBlock(x, 3, 64)

            x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME', name='layer_1')
            x = tf.layers.batch_normalization(x, training=self.training)
            x = x + skip

            for i in range(self.num_upsamples):
                x = self.Upsample2xBlock(x, kernel_size=3, filter_size=256)

            x = tf.nn.conv2d(x, weights['w_out'], strides=[1,1,1,1], padding='SAME', name='y_predict')

            print(x)
            return x

    def _content_loss(self, y, y_pred):
        """MSE, VGG22, or VGG54"""
        if self.content_loss == 'mse':
            return tf.reduce_mean(tf.square(y - y_pred))

        if self.content_loss == 'L1':
            return tf.reduce_mean(tf.abs(y - y_pred))

        if self.content_loss == 'edge_loss_mse':
            y_sobeled = tf.image.sobel_edges(y)
            y_pred_sobeled = tf.sobel_edges(y_pred)
            return tf.reduce_mean(tf.square(y - y_pred)) + tf.reduce_mean(tf.square(y_sobeled - y_pred_sobeled))

        if self.content_loss == 'edge_loss_L1':
            y_sobeled = tf.image.sobel_edges(y)
            y_pred_sobeled = tf.sobel_edges(y_pred)
            return tf.reduce_mean(tf.abs(y - y_pred)) + tf.reduce_mean(tf.square(y_sobeled - y_pred_sobeled))

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