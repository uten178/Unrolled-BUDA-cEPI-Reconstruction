import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Subtract, Lambda
from tensorflow.keras.models import Model

from utils.tensorflow_utils import complex2real, real2complex, fft2c, ifft2c

def KINet_large(data_consistency_type,
         num_rows,
         num_cols,
         num_coils,
         num_GD_blocks=10,
         num_inner_GD_iters=1,
         GD_step_size=0.1,
         activation_type='relu',
         use_DL_regularizer=True,
         pretrained_reg_net_path=None,
         pretrained_net_path=None,
         data_consistency_before_reg=True,
          use_layer_norm=True):

    if pretrained_net_path is not None:
        pretrained_weights_reg_path = None

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

    if use_DL_regularizer:

        from models.UNet2D import UNet2D

        reg_models = {}
        for idx_block in range(num_GD_blocks):
            reg_models[str(idx_block)] = UNet2D(im_size=num_rows, kernel_size=3, num_out_chan_highest_level=64,
                           depth=5, num_chan_increase_rate=2, activation_type=activation_type,
                           dropout_rate=0.05, SEPARABLE_CONV=False, SKIP_CONNECTION_AT_THE_END=True,
                           num_input_chans=2, num_output_chans=2)

            if pretrained_reg_net_path is not None:
                reg_models[str(idx_block)].load_weights(pretrained_reg_net_path)

    x = undersampled_ispace
    Ahb = compute_Ah()(undersampled_kspace, coil_sens, sampling_mask)

    # We define it here (not in the for-loop below) to avoid creating new function every GD iteration
    compute_AhAx_model = getDataTerm_AhAx()

    for idx_block in range(round(num_GD_blocks/2)):

        if data_consistency_before_reg:

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, coil_sens, sampling_mask)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])

                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

            # Include UNet as a regularizer in k-space
            if use_DL_regularizer:

                # image- to k-space
                x = fft2c(x)

                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex2real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = reg_models[str(2*idx_block)](x)

                # Real to complex
                x = real2complex(x)

                # k- to image-space
                x = ifft2c(x)

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, coil_sens, sampling_mask)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])

                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

            # Include UNet as a regularizer in image-space
            if use_DL_regularizer:

                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex2real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = reg_models[str(2*idx_block+1)](x)

                # Real to complex
                x = real2complex(x)

        else:

            # Include UNet as a regularizer in k-space
            if use_DL_regularizer:

                # image- to k-space
                x = fft2c(x)

                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex2real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = reg_models[str(2*idx_block)](x)

                # Real to complex
                x = real2complex(x)

                # k- to image-space
                x = ifft2c(x)

            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, coil_sens, sampling_mask)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])


                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

            # Include UNet as a regularizer in image-space
            if use_DL_regularizer:

                # complex to real: output shape = batch x row x col x 2 (real/imag)
                x = complex2real(x)

                # Pass the images to the regularization network
                if use_layer_norm:
                    x = LayerNormalization(axis=[1,2,3])(x)
                x = reg_models[str(2*idx_block+1)](x)

                # Real to complex
                x = real2complex(x)


            for idx_inner_iter in range(num_inner_GD_iters):
                AhAx = compute_AhAx_model(x, coil_sens, sampling_mask)

                # Compute the gradient of the data consistency term: Ah(A(x)) - Ah(b)
                # GD_grad = AhAx - Ahb  # grad_x 1/2||Ax-b||^2 = Ah(A(x)) - Ah(b)
                GD_grad = Subtract()([AhAx, Ahb])


                # Update the solution: shape = batch x 200 x 200
                # x = x - GD_step_size * GD_grad  # [None, row,col,coil] complex64

                temp = Lambda(lambda x: x * GD_step_size)(GD_grad)
                x = Subtract()([x, temp])

    output_data = complex2real(x)

    model = Model(inputs=[undersampled_ispace, coil_sens, sampling_mask, undersampled_kspace], outputs=output_data)

    if pretrained_net_path is not None:
        model.load_weights(pretrained_net_path)

    return model