# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
# Edited slightly by Brooke Czerwinski to test with 3-channel RGB. Copied from example_dog.py
This script reconstructs a hyperspectral image from a dispersed image using a test color-checker
originally in NEF format into RGB.
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio 

# These functions implements the finite differences method
from finite_differences import *
# Functions for the forward model and their conjugate functions
from forward import *
# Detect edge pixels, convert edge pixels into a vector and back
from edgeOperator import edgeDetect, Mb, Mf
# Functions for evaluating the accuracy of the model
from evaluate_model import spectrumRMSRE

# 3 steps in this hyperspectral image reconstruction algorithm
from step1 import step1
from step2 import step2
from step3 import step3

from dispersionImg import DispersionImg
import os

# Normalize the path for whatever operating system
imgLocation = os.path.normpath('DSC_5984.NEF')
dispersedImgObj = DispersionImg(imgLocation, 512)
# Print the information about the object to the console
dispersedImgObj.printImageInformation()

# Set the rgb_img to the smaller version so it's easier to process than full resolution.
rgb_img = dispersedImgObj.smallerImg

# # Rotate the image for testing
# rgb_img = np.rot90(rgb_img, k=2, axes=(1,0))
# plt.imshow(rgb_img)
# plt.show()

# Use the input RGB image data
j = rgb_img.astype(float) / 255

# h, w, 31 = j.shape # Height, width, and channels
h, w, _ = j.shape
c = 31

# NEXT STEP: Write something to programmatically try different parameter values and combinations for step1
# Step1: Align j into a non-dispersed image
alpha1 = 1e-3
beta1 = 1e-3
rho1 = 1e-2
rho2 = 1e-2
num_iters = 10
cg_iters = 10           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
aligned_hsi = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
                    num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)

# Convert aligned hyperspectral image to RGB image
aligned_img = hsi2rgb(aligned_hsi)


# Display the image that has gone through step1
plt.title('Aligned RGB after step1')
plt.imshow(aligned_img)
plt.show()

# print(aligned_img.shape)
# for matrix in aligned_img:
#     print(matrix)

# Obtain the edge of the aligned image
edge = edgeDetect(aligned_img, ksize=3, sigma=0.5, percentile=92)
E = int(Mf(aligned_hsi, edge).shape[0]) # N

# Step2: Recover the gradient of the hyperspectral image with 
# respect to dispersion direction x 
alpha2 = 1e-3
beta2 = 1e-3
rho3 = 1e-2
num_iters = 10
cg_iters = 10           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
# Find the gradient of the hyperspectral image's edge pixels
vx = step2(j, edge, E, h, w, c, alpha2=alpha2, beta2=beta2, rho3=rho3,
           num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)
# The gradient of the hyperspectral image in x direction
gx = Mb(vx, edge, (h, w, c))

# Step3: Recover hyperspectral image
alpha3 = 2e-2
beta3 = 1e-4
cg_iters = 100           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
reconstructed_hsi = step3(j, gx, h, w, c, alpha3=alpha3, beta3=beta3, 
                          cg_iters=cg_iters, cg_tolerance=cg_tolerance)

# Clip the margin
reconstructed_hsi = reconstructed_hsi[:, :-32, :]

# Convert reconstructed hyperspectral image to RGB image
recovered_img = hsi2rgb(reconstructed_hsi)
recovered_img = np.clip(recovered_img, 0, 1)


# Results
plt.figure()
plt.imshow(j)
plt.axis('off')
plt.savefig("result_czer_test/capturedRGBimage.png", bbox_inches='tight')
plt.savefig("result_czer_test/capturedRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')

plt.figure()
plt.imshow(recovered_img)
plt.axis('off')
plt.savefig("result_czer_test/recoveredRGBimage.png", bbox_inches='tight')
plt.savefig("result_czer_test/recoveredRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')


# PSNR = round(peak_signal_noise_ratio(recovered_img, j),2)
# print(PSNR) # 23.93
