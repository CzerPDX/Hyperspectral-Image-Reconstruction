# -*- coding: utf-8 -*-
"""
@author: Yixuan Shao
# Edited slightly by Brooke Czerwinski to test with 3-channel RGB. Copied from example_dog.py
This script reconstructs a hyperspectral image from a dispersed image using a test color-checker
originally in NEF format into RGB.
"""
import math
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
imgLocation = os.path.normpath('v2_dispersed.NEF')
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

def params_test(j,h,w,c,alph, bet, r1, r2, nit, cgit, cgtol, name):
    num_images=len(alph)*len(bet)*len(r1)*len(r2)*len(nit)*len(cgit)*len(cgtol)
    fighigh=math.floor(math.sqrt(num_images))
    figwide=math.ceil(math.sqrt(num_images))
    table=plt.figure(figsize=(15,10))
    m=1
    for a in alph:
        for b in bet:
            for d in r1:
                for f in r2:
                    for g in nit:
                        for k in cgit:
                            for l in cgtol:
                                aligned_test = step1(j, h, w, c, alpha1=a, beta1=b, rho1=d, rho2=f, 
                                num_iters=g, cg_iters=k, cg_tolerance=l)
                                aligned_testrgb = hsi2rgb(aligned_test)
                                table.add_subplot(figwide,fighigh,m)
                                plt.imshow(aligned_testrgb)
                                plt.axis('off')
                                plt.title('a: %f b: %f r1: %f r2: %f num its: %i cg its: %i cg tol: %f' %(a, b, d, f, g, k, l))
                                m+=1
    plt.savefig(f"result_czer_test/{name}imagetable.png", format='png')
    #plt.show()

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
# plt.title('Aligned RGB after step1')
# plt.imshow(aligned_img)
# plt.show()

alphal=[]
betal=[]
rhol1=[]
rhol2=[]
num_itersl=[]
cg_itersl=[]
cg_tolerancel=[]

""" alphal.extend((1e-3, 1e-4, 1e-5, 1e-6))#alpha1)
betal.extend((1e-3, 1e-4, 1e-5, 1e-6))#beta1)
rhol1.append(rho1)
rhol2.append(rho2)
num_itersl.append(num_iters)
cg_itersl.append(cg_iters)
cg_tolerancel.append(cg_tolerance)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'alpha_beta')

alphal.clear()
betal.clear()
rhol1.clear()
rhol2.clear()

alphal.extend((1e-3, 1e-4, 1e-5, 1e-6))#alpha1)
betal.append(beta1)#extend((1e-3, 1e-4, 1e-5, 1e-6))#beta1)
rhol1.extend((1e-2, 1e-3, 1e-4, 1e-5))#append(rho1)
rhol2.append(rho2)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'alpha_rho1')

alphal.clear()
betal.clear()
rhol1.clear()
rhol2.clear()

alphal.extend((1e-3, 1e-4, 1e-5, 1e-6))#alpha1)
betal.append(beta1)#extend((1e-3, 1e-4, 1e-5, 1e-6))#beta1)
rhol1.append(rho1)
rhol2.extend((1e-2, 1e-3, 1e-4, 1e-5))#append(rho2)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'alpha_rho2')

alphal.clear()
betal.clear()
rhol1.clear()
rhol2.clear()

alphal.append(alpha1)#(1e-3, 1e-4, 1e-5, 1e-6))#alpha1)
betal.extend((1e-3, 1e-4, 1e-5, 1e-6))#beta1)
rhol1.extend((1e-2, 1e-3, 1e-4, 1e-5))#append(rho1)
rhol2.append(rho2)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'beta_rho1')

alphal.clear()
betal.clear()
rhol1.clear()
rhol2.clear()

alphal.append(alpha1)#(1e-3, 1e-4, 1e-5, 1e-6))#alpha1)
betal.extend((1e-3, 1e-4, 1e-5, 1e-6))#beta1)
rhol1.append(rho1)
rhol2.extend((1e-2, 1e-3, 1e-4, 1e-5))#append(rho2)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'beta_rho2')

alphal.clear()
betal.clear()
rhol1.clear()
rhol2.clear()

alphal.append(alpha1)#(1e-3, 1e-4, 1e-5, 1e-6))#alpha1)
betal.append(beta1)#extend((1e-3, 1e-4, 1e-5, 1e-6))#beta1)
rhol1.extend((1e-2, 1e-3, 1e-4, 1e-5))#append(rho1)
rhol2.extend((1e-2, 1e-3, 1e-4, 1e-5))#append(rho2)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rhos')

alphal.clear()
betal.clear()
rhol1.clear()
rhol2.clear()
num_itersl.clear()

alphal.extend((1e-3, 1e-4, 1e-5, 1e-6))
betal.append(beta1)
rhol1.append(rho1)
rhol2.append(rho2)
num_itersl.extend((8,9,10,11,12))
#cg_itersl.append(cg_iters)
#cg_tolerancel.append(cg_tolerance)
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'alphanumits')

num_itersl.clear()
cg_itersl.clear()
num_itersl.append(num_iters)
cg_itersl.extend((8,9,10,11,12))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'alphacgits') """

alphal.extend((1e-3, 1e-4, 1e-5, 1e-6))
betal.append(beta1)
rhol1.append(rho1)
rhol2.append(rho2)
num_itersl.append(num_iters)
cg_itersl.append(cg_iters)
cg_tolerancel.extend((1e-10,1e-11,1e-12,1e-13,1e-14))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'alphacgtol')

alphal.clear()
betal.clear()
alphal.append(alpha1)
betal.extend((1e-3, 1e-4, 1e-5, 1e-6))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'betacgtol')

# cg_tolerancel.clear()
# cg_itersl.clear()
# cg_tolerancel.append(cg_tolerance)
# cg_itersl.extend((8,9,10,11,12))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'betacgits')

# cg_itersl.clear()
# num_itersl.clear()
# cg_itersl.append(cg_iters)
# num_itersl.extend((8,9,10,11,12))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'betanumits')

betal.clear()
rhol1.clear()
betal.append(beta1)
rhol1.extend((1e-2, 1e-3, 1e-4, 1e-5))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rho1numits')

# num_itersl.clear()
# cg_itersl.clear()
# num_itersl.append(num_iters)
# cg_itersl.extend((8,9,10,11,12))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rho1cgits')

# cg_itersl.clear()
# cg_tolerancel.clear()
# cg_itersl.append(cg_iters)
# cg_tolerancel.extend((1e-10,1e-11,1e-12,1e-13,1e-14))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rho1cgtol')

rhol1.clear()
rhol2.clear()
rhol1.append(rho1)
rhol2.extend((1e-2, 1e-3, 1e-4, 1e-5))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rho2cgtol')

# cg_tolerancel.clear()
# cg_itersl.clear()
# cg_tolerancel.append(cg_tolerance)
# cg_itersl.extend((8,9,10,11,12))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rho2cgits')

# cg_itersl.clear()
num_itersl.clear()
# cg_itersl.append(cg_iters)
num_itersl.extend((8,9,10,11,12))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'rho2numits')

# cg_itersl.clear()
rhol2.clear()
rhol2.append(rho2)
# cg_itersl.extend((8,9,10,11,12))
# params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'cgnumits')

# cg_itersl.clear()
# cg_tolerancel.clear()
# cg_itersl.append(cg_iters)
# cg_tolerancel.extend((1e-10,1e-11,1e-12,1e-13,1e-14))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'cgtolnumits')

num_itersl.clear()
cg_itersl.clear()
num_itersl.append(num_iters)
cg_itersl.extend((8,9,10,11,12))
params_test(j, h, w, c, alphal, betal, rhol1, rhol2, num_itersl, cg_itersl, cg_tolerancel,'cgs')

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

# #lines 77-132 are the tests for adjusting iters and tolerance
# i=1
# fig=plt.figure()
# for a in range(9,13,1):
#     num_iters=a
#     cg_iters=a
#     cg_tolerance= 1e-9
#     aligned_1e9 = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
#             num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)
#     aligned_9rgb = hsi2rgb(aligned_1e9)

#     fig.add_subplot(4,4,i)
#     plt.imshow(aligned_9rgb)
#     #plt.show()
#     plt.axis('off')
#     plt.title('Tolerance: 1e-9')
#     i+=1
    
#     cg_tolerance= 1e-10
#     aligned_1e10 = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
#             num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)
#     aligned_10rgb = hsi2rgb(aligned_1e10)

#     fig.add_subplot(4,4,i)
#     plt.imshow(aligned_10rgb)
#     #plt.show()
#     plt.axis('off')
#     plt.title('Tolerance: 1e-10')
#     i+=1
    
#     cg_tolerance= 1e-11
#     aligned_1e11 = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
#             num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)
#     aligned_11rgb = hsi2rgb(aligned_1e11)

#     fig.add_subplot(4,4,i)
#     plt.imshow(aligned_11rgb)
#     #plt.show()
#     plt.axis('off')
#     plt.title('Tolerance: 1e-11')
#     i+=1
    
#     cg_tolerance= 1e-12
#     aligned_1e12 = step1(j, h, w, c, alpha1=alpha1, beta1=beta1, rho1=rho1, rho2=rho2,
#             num_iters=num_iters, cg_iters=cg_iters, cg_tolerance=cg_tolerance)
#     aligned_12rgb = hsi2rgb(aligned_1e12)

#     fig.add_subplot(4,4,i)
#     plt.imshow(aligned_12rgb)
#     #plt.show()
#     plt.axis('off')
#     plt.title('Tolerance: 1e-12')
#     i+=1
# plt.show()
# plt.savefig("result_czer_test/RGBimagetable.png", bbox_inches='tight')
# plt.savefig("result_czer_test/RGBimagetable.svg", dpi=300, format='svg',
#             bbox_inches='tight')
#   It turns out that it will not save as a png, it ends up saving a white box. It does, however, display
