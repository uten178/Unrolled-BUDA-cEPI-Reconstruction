#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:57:31 2022
@author: uten
"""

import tensorflow as tf
import numpy as np
import scipy.signal
import scipy.io as sio
from tensorflow.keras.layers import Lambda

############## U-Net
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Add, \
    LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D, \
    Conv2DTranspose, Dropout, concatenate, SeparableConv2D, PReLU
from tensorflow.keras.models import Model

#################################################################################
################################# UNet ##########################################
#################################################################################
def prelu(_x):
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
    image_out = tf.stack([tmp1, tmp2, tmp3, tmp4, tmp5,tmp6, tmp7, tmp8],axis = -1) 
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
def UNet2D_(im_size, kernel_size=3, num_out_chan_highest_level=64, depth=4, num_chan_increase_rate=2,
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

##############
class BUDA_DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, filename_list,num_rows, num_cols, num_coils, num_polarities, num_time_segs,batch_size,
                 shuffle=True):

        self.filename_list = filename_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.filename_list)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_coils = num_coils
        self.num_polarities = num_polarities
        self.num_time_segs = num_time_segs
        self.on_epoch_end()


    def on_epoch_end(self):
        if self.shuffle:
            #Shuffle the filename list in-place
            np.random.shuffle(self.filename_list)


    def __get_data(self, filenames):

        #Ahb = np.empty((self.batch_size,self.num_rows-8,self.num_cols-8,self.num_polarities))
        csm_buda = np.empty((self.batch_size,self.num_rows,self.num_cols,self.num_coils)).astype(np.csingle)
        mask_buda = np.empty((self.batch_size,self.num_rows,self.num_cols,self.num_polarities,self.num_time_segs)).astype(np.csingle)
        wmap_buda = np.empty((self.batch_size,self.num_rows,self.num_cols,self.num_polarities,self.num_time_segs)).astype(np.csingle)
        kdata_buda = np.empty((self.batch_size,self.num_rows,self.num_cols,self.num_coils,self.num_polarities)).astype(np.csingle)
        label_img = np.empty((self.batch_size,self.num_rows+20,self.num_cols+20,self.num_polarities*2))
        b0_img = np.empty((self.batch_size,self.num_rows+20,self.num_cols+20,self.num_polarities)).astype(np.csingle)

        for idx, curr_filename in enumerate(filenames):
            kdata_buda[idx,], csm_buda[idx,], mask_buda[idx,], wmap_buda[idx,], b0_img[idx,], label_img[idx,] = self.prepare_single_input_output_pair(curr_filename)
        return tuple([kdata_buda,csm_buda,mask_buda,wmap_buda,b0_img]), label_img

    # Return the index'th batch
    def __getitem__(self, index):
        curr_filenames = self.filename_list[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(curr_filenames)

        return X, y

    def __len__(self):
        return self.num_samples // self.batch_size

    def prepare_single_input_output_pair(self,filename_one_sample):
        temp = sio.loadmat(filename_one_sample)
        # Put whatever it needs to generate Ahb, csm, mask, kdata, fully_sampled_ispace
        # for the current sample. For example,
        b0_img = temp['b0'].astype(np.csingle)
        b0_img = b0_img/np.max(np.abs(b0_img))
        label_img = temp['label_img'].astype(np.csingle)
        label_img = label_img/np.max(np.abs(label_img))
        label_img = complex_to_real(label_img)
        kdata_buda = temp['kdata_buda'].astype(np.csingle)
        kdata_buda = kdata_buda/np.max(np.abs(kdata_buda))
        mask_buda = temp['mask_buda'].astype(np.csingle)
        csm_buda = temp['csm_buda'].astype(np.csingle)
        wmap_buda = temp['wmap_buda'].astype(np.csingle)        

        return 3*kdata_buda, csm_buda, mask_buda, wmap_buda, b0_img, label_img

#######################################################################################
fft2c = Lambda(lambda x: fft2c_tf(x))
ifft2c = Lambda(lambda x: ifft2c_tf(x))
fft2c_coil = Lambda(lambda x: fft2c_coil_tf(x))
ifft2c_coil = Lambda(lambda x: ifft2c_coil_tf(x))
fft2c_coil_buda = Lambda(lambda x: fft2c_coil_tf_buda(x))
ifft2c_coil_buda = Lambda(lambda x: ifft2c_coil_tf_buda(x))   

complex_to_real = Lambda(lambda x: complex_to_real_tf(x))
real_to_complex = Lambda(lambda x: real_to_complex_tf(x))

virtual_coil = Lambda(lambda x: virtual_coil_tf(x))
actual_coil = Lambda(lambda x: actual_coil_tf(x))

#######################################################################################   
def normalize(data,batch):
    max_val= tf.math.reduce_max(data,axis=(1, 2, 3))
    norm_data = []
    for it in range(batch):
        tmp = data[it,:,:,:]/max_val[it]
        norm_data.append(tmp)
        
    return tf.stack(norm_data,axis=0)
    
def virtual_coil_tf(image):
    vc_image = tf.transpose(image,(0,2,1,3),conjugate=True)
    image_out = tf.concat((image,vc_image),axis=3) 
    return image_out

def actual_coil_tf(image):
    tmp1 = 0.5*(image[...,0]+tf.transpose(image[...,2],(0,2,1),conjugate=True))
    tmp2 = 0.5*(image[...,1]+tf.transpose(image[...,3],(0,2,1),conjugate=True))
    image_out = tf.stack([tmp1,tmp2],axis = -1) 
    return image_out

def complex_to_real_tf(image):
    image_out = tf.stack([tf.math.real(image), tf.math.imag(image)], axis=-1)
    shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1]*2]],axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out

def real_to_complex_tf(image):
    image_out = tf.reshape(image, [-1, 2])
    image_out = tf.complex(image_out[:, 0], image_out[:, 1])
    shape_out = tf.concat([tf.shape(image)[:-1], [image.shape[-1] // 2]],axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out


def fft2c_tf(x):
    # x: [batch, row, col] ... x in this case
    # tf.signal.fft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    # Inner-most dimension = right-most dimension

    Fx = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2, -1))), axes=(-2, -1))/tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return Fx

def ifft2c_tf(x):
    # x: [batch, row, col] ...k in this case
    # tf.signal.ifft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    Ft_x = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))*tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return Ft_x

# fft2c with coil dimension
def fft2c_coil_tf(x):
    # x: [batch, row, col, coil] ... Cx in this case
    # tf.signal.fft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    # Inner-most dimension = right-most dimension
    # So, we need to swap things around
    x = tf.transpose(x, perm=(0, 3, 1, 2))  # -> [batch, coil, row, col]

    Fx = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2, -1))), axes=(-2, -1))/tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return tf.transpose(Fx, perm=(0, 2, 3, 1))  # -> Back to [batch, row, col, coil]

# ifft2c with coil dimension
def ifft2c_coil_tf(x):
    # x: [batch, row, col, coil] ...Mt_k in this case
    # tf.signal.ifft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    x = tf.transpose(x, perm=(0, 3, 1, 2))  # -> [row,col,coil, batch]
    Ft_x = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))*tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return tf.transpose(Ft_x, perm=(0, 2, 3, 1))  # -> Back to [batch, row, col, coil]

# fft2c with time segmentation & polarities dimension
def fft2c_coil_tf_buda(x):
    # x: [batch, row, col, coil, polar] ... Cx in this case
    # tf.signal.fft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    # Inner-most dimension = right-most dimension
    # So, we need to swap things around
    x = tf.transpose(x, perm=(0, 4, 3, 1, 2))  # -> [batch, polar, coil, row, col]

    Fx = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2, -1))), axes=(-2, -1))/tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return tf.transpose(Fx, perm=(0, 3, 4, 2, 1))  # -> Back to [batch, row, col, coil, polar]

# ifft2c with coil dimension
def ifft2c_coil_tf_buda(x):
    # x: [batch, row, col, coil, polar, tseg] ...Mt_k in this case
    # tf.signal.ifft2d computes the 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of input.
    x = tf.transpose(x, perm=(0, 4, 3, 1, 2))  # -> [row,col,coil, batch]
    Ft_x = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(x, axes=(-2, -1))), axes=(-2, -1))*tf.cast(tf.math.sqrt(float(x.shape[-2]*x.shape[-1])),dtype=tf.complex64)
    return tf.transpose(Ft_x, perm=(0, 3, 4, 2, 1))  # -> Back to [batch, row, col, coil]

def mbir_buda_transpose(kspace, csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities):
    """Apply transpose model.
    k-Space domain to image domain
    """
##
    #num_batchs = kspace.shape[0]
    csm = tf.tile(tf.expand_dims(csm,axis=-1), [1, 1, 1, 1, num_polarities])
    #csm = tf.tile(tf.expand_dims(csm,axis=-1), [1, 1, 1, 1, 1, num_segs])    
    wmap = tf.tile(tf.expand_dims(wmap,axis=3), [1, 1, 1, num_coils, 1, 1])
    mask = tf.tile(tf.expand_dims(mask,axis=3), [1, 1, 1, num_coils, 1, 1])
    #kspace = tf.tile(tf.expand_dims(kspace,axis=-1), [1, 1, 1, 1, 1, num_segs])
##                                        ##############################################################################################
    out = csm[:,:,:,1,:]*0
    for it in range(num_segs):
      tmp_k = kspace*mask[:,:,:,:,:,it]
    #kspace = tf.reshape(kspace,[num_batchs,num_rows,num_cols,num_coils*num_polarities*num_segs])
      tmp_i = ifft2c_coil_buda(tmp_k)
    #tmp_i = tf.reshape(tmp_i,[num_batchs,num_rows,num_cols,num_coils, num_polarities, num_segs])
   
      #out = tf.reduce_sum(tmp_i*tf.math.conj(wmap)*tf.math.conj(csm),axis=5)
      tmp_i = tmp_i*tf.math.conj(wmap[:,:,:,:,:,it])*tf.math.conj(csm)
      out = tf.add(tf.reduce_sum(tmp_i,axis=3),out)

    out = tf.image.resize_with_crop_or_pad(out,num_rows+20,num_rows+20) ###########################################################

    return out

def mbir_buda_forward(img, csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities):
    """Apply forward model.
    k-Space domain to image domain
    """
    img = tf.image.resize_with_crop_or_pad(img,num_rows,num_cols)
##
    csm = tf.tile(tf.expand_dims(csm,axis=-1), [1, 1, 1, 1, num_polarities])
    #csm = tf.tile(tf.expand_dims(csm,axis=-1), [1, 1, 1, 1, 1, num_segs])    
    wmap = tf.tile(tf.expand_dims(wmap,axis=3), [1, 1, 1, num_coils, 1, 1])
    mask = tf.tile(tf.expand_dims(mask,axis=3), [1, 1, 1, num_coils, 1, 1])
    img = tf.tile(tf.expand_dims(img,axis=3), [1, 1, 1, num_coils, 1])
    #img = tf.tile(tf.expand_dims(img,axis=-1), [1, 1, 1, 1, 1, num_segs])
##
    out = csm*0
    for it in range(num_segs):
      tmp_i = img*wmap[:,:,:,:,:,it]*csm
    #img = tf.reshape(img,[num_batchs, num_rows, num_cols, num_coils*num_polarities*num_segs])
      tmp_k = fft2c_coil_buda(tmp_i)
    #tmp_k = tf.reshape(tmp_k,[num_batchs, num_rows, num_cols, num_coils, num_polarities, num_segs])
      tmp_k = tmp_k*mask[:,:,:,:,:,it]
      out = tf.add(out,tmp_k)#tf.reduce_sum(tmp_k,axis=5)

    return out

class compute_Ah(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(compute_Ah, self).__init__()
        self.mbir_buda_transpose = mbir_buda_transpose

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(compute_Ah, self).build(input_shape)

    # call, where you do the forward computation
    def call(self, x, csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities):
        # inputs: x, C, M, W

        # Compute Ah(A(x))
        return self.mbir_buda_transpose(x, csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities)


class getDataTerm_AhAx(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(getDataTerm_AhAx, self).__init__()

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(getDataTerm_AhAx, self).build(input_shape)

    # call, where you do the forward computation
    def call(self, x, csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities):
        # inputs: x, C, M, W

        # Compute Ah(A(x))
        return mbir_buda_transpose(mbir_buda_forward(x, csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities), csm, mask, wmap, num_rows, num_cols, num_coils, num_segs, num_polarities)

###################################
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Subtract, Lambda, LayerNormalization
from tensorflow.keras.models import Model


def KINet(num_rows,
         num_cols,
         num_coils,
         num_segs,
         num_polarities,
         num_GD_blocks,
         Batch_size,
         num_inner_GD_iters,
         GD_step_size=0.9,
         activation_type='prelu',
         use_DL_regularizer=True,
         pretrained_unrolled_net_path=None,
         data_consistency_before_reg=True,
         use_layer_norm=False):


## All Inputs
    csm_buda = Input(shape=(num_rows, num_cols, num_coils), dtype=tf.complex64)
    mask_buda = Input(shape=(num_rows, num_cols, num_polarities, num_segs), dtype=tf.complex64)
    kdata_buda = Input(shape=(num_rows, num_cols,num_coils, num_polarities), dtype=tf.complex64)
    wmap_buda = Input(shape=(num_rows, num_cols,num_polarities, num_segs), dtype=tf.complex64)
    b0_img = Input(shape=(num_rows+20, num_cols+20,num_polarities), dtype=tf.complex64)
##
      
    if use_DL_regularizer:     

        regI_models = {}
        regK_models = {}
        for idx_block in range(round(num_GD_blocks/2)):
            regI_models[str(idx_block)] = UNet2D_(im_size=num_rows+20, kernel_size=3, num_out_chan_highest_level=64,
                           depth=3, num_chan_increase_rate=2, activation_type=activation_type,
                           dropout_rate=0.1, SEPARABLE_CONV=False, SKIP_CONNECTION_AT_THE_END=True,
                           num_input_chans=16, num_output_chans=8)           
            regK_models[str(idx_block)] = UNet2D_(im_size=num_rows+20, kernel_size=3, num_out_chan_highest_level=64,
                           depth=3, num_chan_increase_rate=2, activation_type=activation_type,
                           dropout_rate=0.1, SEPARABLE_CONV=False, SKIP_CONNECTION_AT_THE_END=True,
                           num_input_chans=16, num_output_chans=8)
    
########################################################################################################################
    b0_i = virtual_coil(b0_img)
    b0_k = fft2c_coil(b0_i)
    b0_i = complex_to_real(b0_i)
    b0_k = complex_to_real(b0_k)
########################################################################################################################
    x = compute_Ah()(kdata_buda, csm_buda, mask_buda, wmap_buda, num_rows, num_cols, num_coils, num_segs, num_polarities)  
    Ahb = x
    compute_AhAx_model = getDataTerm_AhAx()

    for idx_block in range(round(num_GD_blocks/2)):

        if data_consistency_before_reg:

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, csm_buda, mask_buda, wmap_buda, num_rows, num_cols, num_coils, num_segs, num_polarities)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])

                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

            # Include UNet as a regularizer in k-space
            if use_DL_regularizer:

                x = virtual_coil(x) ## complex conjugate channel ... actual & virtual up & down ... 4 channels
                # image- to k-space
                x = fft2c_coil(x)

                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex_to_real(x) ## complex conjugate channel ... actual & virtual up & down .. real & imaginary ... 8 channels
                
                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = regK_models[str(idx_block)](tf.concat(axis=3, values=[x, b0_k]))

                # Real to complex              
                x = real_to_complex(x)
            
                # k- to image-space
                x = ifft2c_coil(x)
                x = actual_coil(x) ## back to actual up&down complex number ... 2 channels

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, csm_buda, mask_buda, wmap_buda, num_rows, num_cols, num_coils, num_segs, num_polarities)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])

                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

            # Include UNet as a regularizer in image-space
            if use_DL_regularizer:
                x = virtual_coil(x)
                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex_to_real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = regI_models[str(idx_block)](tf.concat(axis=3, values=[x, b0_i]))

                # Real to complex
                x = real_to_complex(x)
                x = actual_coil(x)
                   
        else:

            # Include UNet as a regularizer in k-space
            if use_DL_regularizer:
                x = virtual_coil_tf(x)
                # image- to k-space
                x = fft2c_coil(x)

                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex_to_real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = regK_models[str(idx_block)](tf.concat(axis=3, values=[x, b0_k]))

                # Real to complex
                x = real_to_complex(x)

                # k- to image-space
                x = ifft2c_coil(x)
                x = actual_coil(x)

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, csm_buda, mask_buda, wmap_buda, num_rows, num_cols, num_coils, num_segs, num_polarities)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])


                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

            # Include UNet as a regularizer in image-space
            if use_DL_regularizer:
                x = virtual_coil(x)
                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex_to_real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = regI_models[str(idx_block)](tf.concat(axis=3, values=[x, b0_i]))

                # Real to complex
                x = real_to_complex(x)
                x = actual_coil(x)

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, csm_buda, mask_buda, wmap_buda, num_rows, num_cols, num_coils, num_segs, num_polarities)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])


                # Update the solution: shape = batch x    200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])
    
    output_data = complex_to_real(x)            
    output_data = normalize(output_data,Batch_size)## unit normalization
    model = Model(inputs=[kdata_buda, csm_buda, mask_buda, wmap_buda, b0_img], outputs=output_data)    

    return model
