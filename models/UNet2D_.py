import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Add, \
    LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D, \
    Conv2DTranspose, Dropout, concatenate, SeparableConv2D, PReLU
from tensorflow.keras.models import Model

##
#################################################################################
################################# UNet ##########################################
#################################################################################
def prelu(_x, a=0.2):
  # Parametric ReLU, where a is a variable to be learned.
  alphas =  tf.compat.v1.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

#######################################################################################
def deconcat_data(image):
    tmp1 = image[:,:,:,0]
    tmp2 = image[:,:,:,1]
    tmp3 = image[:,:,:,2]
    tmp4 = image[:,:,:,3]
    tmp5 = image[:,:,:,4]
    tmp6 = image[:,:,:,5]
    tmp7 = image[:,:,:,6]
    tmp8 = image[:,:,:,7]
    tmp9 = image[:,:,:,8]
    tmp10 = image[:,:,:,9]
    tmp11 = image[:,:,:,10]
    tmp12 = image[:,:,:,11]
    tmp13 = image[:,:,:,12]
    tmp14 = image[:,:,:,13]
      
    image_out = tf.stack([tmp1, tmp2, tmp3, tmp4, tmp5,tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14],axis = -1) 
    return image_out  
############################################################################  
# Batch Norm -> Conv2D -> Nonlinearity
def conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', SEPARABLE_CONV=False):
    x = BatchNormalization()(x)

    if SEPARABLE_CONV:
        x = SeparableConv2D(num_out_chan, (kernel_size, kernel_size), activation=None, padding='same',
                            kernel_initializer='truncated_normal')(x)
    else:
        x = Conv2D(num_out_chan, (kernel_size, kernel_size), activation=None, padding='same',
                   kernel_initializer='truncated_normal')(x)

    if activation_type == 'lrelu':
        return LeakyReLU()(x)
    elif activation_type == 'prelu':
        return prelu(x)#PReLU()(x)
    else:
        return Activation(activation_type)(x)


# Batch Norm -> Conv2Dt -> Nonlinearity
def conv2Dt_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu'):
    x = BatchNormalization()(x)

    x = Conv2DTranspose(num_out_chan, (kernel_size, kernel_size), strides=(2, 2), padding='same',
                        kernel_initializer='truncated_normal')(x)

    if activation_type == 'lrelu':
        return LeakyReLU()(x)
    elif activation_type == 'prelu':
        return prelu(x)#PReLU()(x)
    else:
        return Activation(activation_type)(x)


def createOneLevel_UNet2D(x, num_out_chan, kernel_size, depth, num_chan_increase_rate, activation_type, dropout_rate,
                          SEPARABLE_CONV):
    if depth > 0:

        # Left
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type,
                                SEPARABLE_CONV=SEPARABLE_CONV)
        x = Dropout(dropout_rate)(x)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type,
                                SEPARABLE_CONV=SEPARABLE_CONV)
        x = Dropout(dropout_rate)(x)

        x_to_lower_level = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last')(
            x)

        # Lower level
        x_from_lower_level = createOneLevel_UNet2D(x_to_lower_level, int(num_chan_increase_rate * num_out_chan),
                                                   kernel_size, depth - 1, num_chan_increase_rate, activation_type,
                                                   dropout_rate, SEPARABLE_CONV)

        x_conv2Dt = conv2Dt_bn_nonlinear(x_from_lower_level, num_out_chan, kernel_size, activation_type=activation_type)

        # Right
        x = concatenate([x, x_conv2Dt], axis=3)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type,
                                SEPARABLE_CONV=SEPARABLE_CONV)
        x = Dropout(dropout_rate)(x)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type,
                                SEPARABLE_CONV=SEPARABLE_CONV)
        x = Dropout(dropout_rate)(x)

    else:
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type,
                                SEPARABLE_CONV=SEPARABLE_CONV)
        x = Dropout(dropout_rate)(x)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type,
                                SEPARABLE_CONV=SEPARABLE_CONV)
        x = Dropout(dropout_rate)(x)

    return x


# num_out_chan_highest_level = number of filters used for each convolutional layer of the highest U-Net level (i.e., the level with largest spatial dimensions)
# num_input_chans = # of input channels
# num_output_chans = # of output channels
def UNet2D(im_size, kernel_size=3, num_out_chan_highest_level=64, depth=4, num_chan_increase_rate=2,
           activation_type='prelu', dropout_rate=0.05, SEPARABLE_CONV=False, SKIP_CONNECTION_AT_THE_END=False,
           num_input_chans=2, num_output_chans=1):

    input_img = Input(shape=(im_size, im_size, num_input_chans))

    x = conv2D_bn_nonlinear(input_img, num_out_chan_highest_level, kernel_size=5, activation_type=activation_type,
                            SEPARABLE_CONV=SEPARABLE_CONV)

    temp = createOneLevel_UNet2D(x, num_out_chan_highest_level, kernel_size, depth - 1, num_chan_increase_rate,
                                 activation_type, dropout_rate, SEPARABLE_CONV)

    output_img = Conv2D(num_output_chans, (kernel_size, kernel_size), activation=None, padding='same',
                        kernel_initializer='truncated_normal')(temp)

    if SKIP_CONNECTION_AT_THE_END:
        output_img = Add()([deconcat_data(input_img), output_img])

    return Model(inputs=input_img, outputs=output_img)
