# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
This script evaluates the impact of using different optimization parameters
alpha2 and beta2 using the example of a colorchecker
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

# scanning different choice of alpha2 and beta2
alpha2_list = [1e-5, 1e-4, 1e-3]
beta2_list = [1e-2, 1e-1, 1]

averageRMSREs = np.zeros((3,3))
for ii in range(3):
    for jj in range(3):
        # Read the ground truth hyperspectral image data
        name = 'stuffed_toys_ms/stuffed_toys_ms_'
        raw = np.zeros((512, 512, 31))
        for i in range(31):
            if i<9:
                s = "0"+str(i+1)
            else:
                s = str(i+1)
            
            img = io.imread(f'{name}{s}.png').astype(float) / 255 / 255 * 8
            raw[...,i] = img
            
        # Use the colorchecker region
        raw = raw[256:, 100:356, :]
        raw[:, -32:, :] = 0 # Due to dispersion, some of the pixels' spectral information is lost
        h, w, c = raw.shape # Height, width, and channels
    
        # Forward model. 
        # Generate the dispersed image, which is captured by the camera
        j = forward(raw)
        j += np.random.randn(h, w, 3) * np.mean(j) * 0.1
    
        
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
        # Obtain the edge of the aligned image
        edge = edgeDetect(aligned_img, ksize=3, sigma=0.5, percentile=92)
        E = Mf(aligned_hsi, edge).shape[0] # Number of pixels in the edge
        
        # Step2: Recover the gradient of the hyperspectral image with 
        # respect to dispersion direction x 
        alpha2 = alpha2_list[ii]
        beta2 = beta2_list[jj]
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
        alpha3 = 5e-4
        beta3 = 2e-3
        cg_iters = 100           # number of iterations for CG solver
        cg_tolerance = 1e-12    # convergence tolerance of cg solver
        reconstructed_hsi = step3(j, gx, h, w, c, alpha3=alpha3, beta3=beta3, 
                                  cg_iters=cg_iters, cg_tolerance=cg_tolerance)
        
        # Clip the margin
        raw = raw[:, :-32, :]
        reconstructed_hsi = reconstructed_hsi[:, :-32, :]
        
        
        # Blue patch
        y_range = (113, 123)
        x_range = (70, 80)
        RMSRE1, reconstructed_spectrum, groundTruth_spectrum = spectrumRMSRE(reconstructed_hsi, raw, 
                                            y_range=y_range, x_range=x_range)
        
        # Green patch
        y_range = (157, 167)
        x_range = (72, 82)
        RMSRE2, reconstructed_spectrum, groundTruth_spectrum = spectrumRMSRE(reconstructed_hsi, raw, 
                                            y_range=y_range, x_range=x_range)
        
        # White patch
        y_range = (207, 217)
        x_range = (30, 40)
        RMSRE3, reconstructed_spectrum, groundTruth_spectrum = spectrumRMSRE(reconstructed_hsi, raw, 
                                            y_range=y_range, x_range=x_range)
        
        # Orange patch
        y_range = (113, 123)
        x_range = (28, 38)
        RMSRE4, reconstructed_spectrum, groundTruth_spectrum = spectrumRMSRE(reconstructed_hsi, raw, 
                                            y_range=y_range, x_range=x_range)
        
        averageRMSREs[ii, jj] = np.mean([RMSRE1,RMSRE2,RMSRE3,RMSRE4])
    

print(averageRMSREs)
# [[0.17318885 0.17255054 0.17004673]
#  [0.17592881 0.1686397  0.17280327]
#  [0.17228039 0.17266907 0.17001838]]


