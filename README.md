# Unrolled U-Net for BUDA cEPI dMRI Reconstruction
MRI reconstruction with unrolled U-Net as priors
# For Training the model
Train_BUDA_cEPI.py
# For Testing the model
Test_BUDA_cEPI.py
# Pre-trained model
/trained_weights/KI_UNET.hdf5
# Example data for model training and testing
one slice raw k-space data : 300x300x12x2 (Nx x Ny x Ncoil x 2 polarities (Up&Down))
/Train_DATA/xxx.mat\
/Validate_DATA/xxx.mat\
/Test_DATA/xxx.mat
