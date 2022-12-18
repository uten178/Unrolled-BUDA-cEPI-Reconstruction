import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Subtract, Lambda, LayerNormalization, Activation,BatchNormalization,\
    Conv2D, Dropout,LeakyReLU,PReLU
from tensorflow.keras.models import Model

from utils.tensorflow_utils import complex2real, real2complex

def VNet(data_consistency_type,
         num_rows,
         num_cols,
         num_coils,
         num_GD_blocks=10,
         GD_step_size=0.1,
         activation_type='relu',
         num_filters=64,
         dropout_rate=0.1,
         pretrained_net_path=None,
         use_layer_norm=True):


    if data_consistency_type == 'sense':
        from models.forward_backward_ops.SENSE import compute_Ah, getDataTerm_AhAx
    elif data_consistency_type == 'sense-buda':
        from models.forward_backward_ops.SENSE_BUDA import compute_Ah, getDataTerm_AhAx
    else:
        print(data_consistency_type, 'is not currently supported.')
        sys.exit(1)

    #### Create a model
    undersampled_ispace = Input(shape=(num_rows, num_cols), dtype=tf.complex64)  # undersampled (image space)
    coil_sens = Input(shape=(num_rows, num_cols, num_coils), dtype=tf.complex64)
    sampling_mask = Input(shape=(num_rows, num_cols, num_coils), dtype=tf.complex64)
    undersampled_kspace = Input(shape=(num_rows, num_cols, num_coils), dtype=tf.complex64)


    x = undersampled_ispace
    Ahb = compute_Ah()(undersampled_kspace, coil_sens, sampling_mask)

    # We define it here (not in the for-loop below) to avoid creating new function every GD iteration
    compute_AhAx_model = getDataTerm_AhAx()

    for idx_block in range(num_GD_blocks):

        #### Data consistency
        AhAx = compute_AhAx_model(x, coil_sens, sampling_mask)

        # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
        # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
        GD_grad = Subtract()([AhAx, Ahb])
        data_term_update = Lambda(lambda x: x * GD_step_size)(GD_grad)

        #### Regularization
        # complex to real: output shape = batch x row x col x 2 (real/imag)
        x_c2r = complex2real(x)

        # Pass the images to the regularization network
        if use_layer_norm:
            x_c2r = LayerNormalization(axis=[1,2,3])(x_c2r)
        reg_term_update = conv2D_VN(x_c2r, kernel_size=3, activation_type=activation_type, dropout_rate=dropout_rate,
                             num_filters=num_filters)
        reg_term_update = conv2D_VN(reg_term_update, kernel_size=3, activation_type=activation_type, dropout_rate=dropout_rate,
                             num_filters=num_filters)
        reg_term_update = conv2D_VN(reg_term_update, kernel_size=3, activation_type=activation_type, dropout_rate=dropout_rate,
                             num_filters=2)
        # Real to complex
        reg_term_update = real2complex(reg_term_update)

        data_reg_update = Add()([data_term_update, reg_term_update])
        # data_reg_update = Add()([data_term_update, multiplyTrainbleVar()(reg_term_update)])

        #### Update x
        x = Subtract()([x, data_reg_update])

    output_data = complex2real(x)

    model = Model(inputs=[undersampled_ispace, coil_sens, sampling_mask, undersampled_kspace], outputs=output_data)

    if pretrained_net_path is not None:
        model.load_weights(pretrained_net_path)

    return model

def conv2D_VN(x, kernel_size, activation_type, dropout_rate, num_filters):
    # Batch norm
    x = BatchNormalization()(x)

    # 3x3 Conv2D
    x = Conv2D(num_filters, kernel_size, activation=None, use_bias=True, padding='same',
               kernel_initializer='truncated_normal')(x)

    # Activation
    if activation_type == 'lrelu':
        x = LeakyReLU()(x)
    elif activation_type == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)

    # Dropout
    return Dropout(dropout_rate)(x)


class multiplyTrainbleVar(tf.keras.layers.Layer):
    def __init__(self, **kwargs):

        super(multiplyTrainbleVar, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.lambdaa = self.add_weight(name='lambda',
                                  shape=(1,),
                                  initializer='uniform',
                                  trainable=True)

        # Debugging: Uncomment to set lambda to zero
        '''self.lambdaa = self.add_weight(name='lambda',
                                  shape=(1,),
                                  initializer='zeros',
                                  trainable=False)'''

        super(multiplyTrainbleVar, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #return tf.multiply(x,self.lambdaa)
        return tf.multiply(x,tf.keras.backend.abs(self.lambdaa))

    def compute_output_shape(self, input_shape):
        return input_shape