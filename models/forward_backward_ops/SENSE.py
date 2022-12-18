import tensorflow as tf
from utils.tensorflow_utils import fft2c_coil, ifft2c_coil


def applyA(x, coil_sens, sampling_mask):
    # x (image space): [None, 128, 128]
    # coil_sens: [None, 128, 128, 8]
    # sampling_mask: [None, 128, 128, 8]

    Cx = coil_sens * tf.expand_dims(x, axis=3)
    return sampling_mask * fft2c_coil(Cx)


def applyA_transpose(kspace, coil_sens, sampling_mask):
    # kspace: [None,128,128,8]
    # coil_sens: [None, 128, 128, 8]
    # sampling_mask: [None, 128, 128, 8]

    FtMt_k = ifft2c_coil(tf.math.conj(sampling_mask) * kspace)
    CtFtMt_k = tf.math.conj(coil_sens) * FtMt_k
    return tf.math.reduce_sum(CtFtMt_k, axis=3)


class compute_Ah(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(compute_Ah, self).__init__()
        self.applyA_transpose = applyA_transpose

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(compute_Ah, self).build(input_shape)

    # call, where you do the forward computation
    def call(self, x, coil_sens, sampling_mask):
        # inputs: x, C, M

        # Compute Ah(A(x))
        return self.applyA_transpose(x, coil_sens, sampling_mask)


class getDataTerm_AhAx(tf.keras.layers.Layer):

    # __init__ , where you can do all input-independent initialization
    def __init__(self):
        super(getDataTerm_AhAx, self).__init__()

    # # build, where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        super(getDataTerm_AhAx, self).build(input_shape)

    # call, where you do the forward computation
    def call(self, x, coil_sens, sampling_mask):
        # inputs: x, C, M

        # Compute Ah(A(x))
        return applyA_transpose(applyA(x, coil_sens, sampling_mask), coil_sens, sampling_mask)

