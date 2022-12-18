import tensorflow as tf
from tensorflow.keras.layers import Lambda

complex2real = Lambda(lambda x: tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1))

real2complex = Lambda(lambda x: tf.complex(x[..., 0], x[..., 1]))

fft2c = Lambda(lambda x: fft2c_tf(x))
ifft2c = Lambda(lambda x: ifft2c_tf(x))
fft2c_coil = Lambda(lambda x: fft2c_coil_tf(x))
ifft2c_coil = Lambda(lambda x: ifft2c_coil_tf(x))

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