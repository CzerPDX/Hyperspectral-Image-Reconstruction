# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
This script reconstructs a hyperspectral image from a dispersed image
using the example of a toy dog
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
dispersedImgObj = DispersionImg(imgLocation)
# Print the information about the object to the console
dispersedImgObj.printImageInformation()

# Set the rgb_image to the smaller version so it's easier to process than full resolution.
rgb_image = dispersedImgObj.smallerImg

# Use the input RGB image data
j = rgb_image.astype(float) / 255

h, w, c = j.shape # Height, width, and channels

# Step1: Align j into a non-dispersed image
alpha1 = 1e-3
beta1 = 1e-3
rho1 = 1e-2
rho2 = 1e-2
num_iters = 10
cg_iters = 10           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
aligned_rgb = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
                    num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)

# Obtain the edge of the aligned image
edge = edgeDetect(aligned_rgb, ksize=3, sigma=0.5, percentile=92)
E = edge.sum() # Number of pixels in the edge

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
gx = Mb(vx, edge)

# Step3: Recover hyperspectral image
alpha3 = 2e-2
beta3 = 1e-4
cg_iters = 100           # number of iterations for CG solver
cg_tolerance = 1e-12    # convergence tolerance of cg solver
reconstructed_rgb = step3(j, gx, h, w, c, alpha3=alpha3, beta3=beta3, 
                          cg_iters=cg_iters, cg_tolerance=cg_tolerance)



recovered_img = hsi2rgb(rgb_image)


# Results
plt.figure()
plt.imshow(j)
plt.axis('off')
plt.savefig("result_toy/capturedRGBimage.png", bbox_inches='tight')
plt.savefig("result_toy/capturedRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')

plt.figure()
plt.imshow(recovered_img)
plt.axis('off')
plt.savefig("result_toy/recoveredRGBimage.png", bbox_inches='tight')
plt.savefig("result_toy/recoveredRGBimage.svg", dpi=300, format='svg',
            bbox_inches='tight')


PSNR = round(peak_signal_noise_ratio(recovered_img, j),2)
print(PSNR) # 23.93
