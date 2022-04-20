#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:33:39 2021

@author: uten yarach
"""
from utils.buda_utils import fft2c, ifft2c
import tensorflow as tf

def mbir_buda_transpose(kspace, csm, mask, wmap):
    """Apply transpose model.
    k-Space domain to image domain
    """
    Nx,Ny,Nc,Np,Nslice = csm.shape
    Nseg = wmap.shape[4]

    X = []
    for i in range(Nseg):
        tmp_k = mask[:,:,:,:,:,i]*kspace
        tmp_k = tf.reshape(tmp_k,[Nx,Ny,Nc*Np*Nslice])
        tmp_i = ifft2c(tmp_k)
        tmp_i = tf.reshape(tmp_i,[Nx,Ny,Nc,Np,Nslice])*tf.math.conj(csm)
        tmp_i = tf.reduce_sum(tmp_i, axis=2)
        X.append(tmp_i)

    X = tf.stack(X,-1)
    out = tf.reduce_sum(X*tf.math.conj(wmap),axis=4)
    out = tf.reshape(out,[Nx,Ny,Np*Nslice])
    return tf.transpose(out,perm=[2, 0, 1])

def mbir_buda_forward(img, csm, mask, wmap):
    """Apply forward model.
    image domain to k-Space
    """
    Nx,Ny,Nc,Np,Nslice = csm.shape
    Nseg = wmap.shape[4]

    img = tf.transpose(img,perm=[1,2,0])
    img = tf.reshape(img,[Nx,Ny,Np,Nslice])
    img = tf.tile(tf.expand_dims(img,axis=4), [1, 1, 1, 1, Nseg])*wmap
    out = tf.zeros([Nx,Ny,Nc,Np,Nslice], tf.complex64)

    for i in range(Nseg):
        tmp_i = img[:,:,:,:,i]
        tmp_i = tf.tile(tf.expand_dims(tmp_i,axis=2), [1, 1, Nc, 1, 1])
        tmp_i = tmp_i*csm
        tmp_i = tf.reshape(tmp_i,[Nx,Ny,Nc*Np*Nslice])
        tmp_k = fft2c(tmp_i)
        tmp_k = tf.reshape(tmp_k,[Nx,Ny,Nc,Np,Nslice])
        tmp_k = tmp_k*mask[:,:,:,:,:,i]
        out = tf.add(tmp_k,out)

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
    def call(self, x, csm, mask, wmap):
        # inputs: x, C, M, W

        # Compute Ah(A(x))
        return self.mbir_buda_transpose(x, csm, mask, wmap)


class getDataTerm_AhAx(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(getDataTerm_AhAx, self).__init__()

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(getDataTerm_AhAx, self).build(input_shape)

    # call, where you do the forward computation
    def call(self, x, csm, mask, wmap):
        # inputs: x, C, M, W

        # Compute Ah(A(x))
        return mbir_buda_transpose(mbir_buda_forward(x, csm, mask, wmap), csm, mask, wmap)