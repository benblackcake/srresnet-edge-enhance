
import tensorflow as tf


class Srresnet:
    """Srresnet Model"""

    def __init__(self, training, content_loss='mse', learning_rate=1e-4, num_blocks=16, num_upsamples=2):
        self.learning_rate = learning_rate
        self.num_blocks = num_blocks
        self.num_upsamples = num_upsamples
        self.training = training

        if content_loss not in ['mse', 'L1']:
            print('Invalid content loss function. Must be \'mse\', or \'L1_loss\'.')
            exit()
        self.content_loss = content_loss

    def ResidualBlock(self, x, kernel_size, filters, strides=1):
        """Residual block a la ResNet"""
        skip = x
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same',
                             use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same',
                             use_bias=False)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = x + skip
        return x

    def Upsample2xBlock(self, x, kernel_size, filters, strides=1):
        """Upsample 2x via SubpixelConv"""
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
        x = tf.depth_to_space(x, 2)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        return x

    def forward(self, x):
        """Builds the forward pass network graph"""
        with tf.variable_scope('srresnet_edge') as scope:
            x = tf.layers.conv2d(x, kernel_size=9, filters=64, strides=1, padding='same')
            x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
            skip = x

            # B x ResidualBlocks
            for i in range(self.num_blocks):
                x = self.ResidualBlock(x, kernel_size=3, filters=64, strides=1)

            x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
            x = tf.layers.batch_normalization(x, training=self.training)
            x = x + skip

            # Upsample blocks
            for i in range(self.num_upsamples):
                x = self.Upsample2xBlock(x, kernel_size=3, filters=256)

            x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')
            return x

    def _content_loss(self, y, y_pred):
        """MSE, VGG22, or VGG54"""
        if self.content_loss == 'mse':
            return tf.reduce_mean(tf.square(y - y_pred))
        if self.content_loss == 'L1':
            return tf.reduce_mean(tf.abs(y - y_pred))


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