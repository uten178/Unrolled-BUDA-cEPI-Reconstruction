import tensorflow as tf
import numpy as np
import scipy.signal
import scipy.io as sio
import time, pdb, os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda

# Deep learning packages
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

##############################################################
from models.UNet2D_ import UNet2D_
from utils.buda_tensorflow_utils import KINet
from utils.buda_tensorflow_utils import BUDA_DataGenerator
from utils.buda_tensorflow_utils import real_to_complex 
#############################################################

data_path = '.../'
sys.path.insert(0,data_path)

########################################################################################################################
num_rows = 300
num_cols = 300
num_coils = 12
num_time_segs = 50
num_polarities = 2
num_batchs = 1  

########################################################################################################################    
input_paths = glob.glob(os.path.join('Train_DATA/*.mat'))              
training_gen = BUDA_DataGenerator(input_paths,num_rows, num_cols, num_coils, num_polarities, num_time_segs,num_batchs,shuffle=True)

####################################################################################
input_paths = glob.glob(os.path.join('Validate_DATA/*.mat'))     
val_gen = BUDA_DataGenerator(input_paths,num_rows, num_cols, num_coils, num_polarities, num_time_segs,num_batchs,shuffle=True)

########################################################################################################################
# "Important" parameters
model_type = 'ki-net' # Options: 'unet', 'inet' 
num_GD_blocks = 6
num_inner_GD_iters = 1
GD_step_size = 0.9 # Step size for the data-consistency-gradient update
data_consistency_type = 'sense-buda'    
data_consistency_before_reg = True # For 'inet', 'knet', 'kikinet'
use_DL_regularize = True
# Deep learning parameters
num_epochs = 500
loss = 'nrmse'
activation_type = 'prelu'
learning_rate_base = 1e-4
batch_size = num_batchs 

tensorboard_filepath = os.path.join(data_path,'results')
model_checkpoint_filepath = os.path.join(data_path,'trained_weights', 'KI_UNET.hdf5')

pre_train_path = ""
pretrained_reg_net_path = ""
pretrained_unrolled_net_path = os.path.join(data_path,'trained_weights', 'KI_UNET.hdf5')

if not os.path.exists(pretrained_reg_net_path):
    pretrained_reg_net_path = None
   
if not os.path.exists(pretrained_unrolled_net_path):
    pretrained_unrolled_net_path = None

##############################################################3
if model_type == 'ki-net':
    model = KINet(data_consistency_type,
                 num_rows,
                 num_cols,
                 num_coils,
                 num_time_segs,
                 num_polarities,
                 batch_size,
                 num_GD_blocks,
                 num_inner_GD_iters,
                 GD_step_size,
                 activation_type,
                 use_DL_regularizer=True,
                 pretrained_reg_net_path=pretrained_reg_net_path,
                 pretrained_inet_path=pretrained_unrolled_net_path,
                 data_consistency_before_reg=data_consistency_before_reg,
                 use_layer_norm = False)
 
elif model_type == 'unet':
    model = UNet2D_(im_size=num_rows, kernel_size=3, num_out_chan_highest_level=64, 
                   depth=5, num_chan_increase_rate=2, activation_type=activation_type, 
                   dropout_rate=0.05,SEPARABLE_CONV=False, SKIP_CONNECTION_AT_THE_END=True, 
                   num_input_chans=4, num_output_chans=4)
    
    if pretrained_reg_net_path is not None:
        model.load_weights(pretrained_reg_net_path)

####################################################################
# Compile the model
adam_opt = Adam(learning_rate=learning_rate_base, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-3)
tbCallBack = TensorBoard(log_dir=tensorboard_filepath, histogram_freq=0, write_graph=False, write_images=False)
checkpointerCallBack = ModelCheckpoint(filepath=model_checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

if loss == 'nrmse':
    # Objective function
    def my_objective_function(y_true, y_pred):
        return 100 * K.sqrt(K.sum(K.square(y_pred - y_true))) / K.sqrt(K.sum(K.square(y_true)))

    model.compile(loss=my_objective_function, optimizer=adam_opt)
else:
    model.compile(loss=loss, optimizer=adam_opt)
    
########################################################################################################################
# model fitting
hist = model.fit(x=training_gen,
                                 epochs=num_epochs,
                                 verbose=1,
                                 validation_data=val_gen,                               
                                 callbacks=[lr_decay, modelcp],
                                 initial_epoch=0)
