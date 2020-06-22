# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:12:18 2020

@author: haris
"""

# import matplotlib.pyplot as plt
# import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat 
# from skimage import data
# from skimage.util import img_as_float
from skimage.io import imread
from skimage.io import imshow
from skimage.io import imsave
# from skimage.filters import gabor_kernel
import scipy.io as sio
from IPython.display import Image, display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import cv2
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from spectral import*

# Read RGB image into an array
img = imread('1.png',as_gray=True, plugin=None)
img10 = imread('10.png',as_gray=True, plugin=None)
img30 = imread('30.png',as_gray=True, plugin=None)
img_shape = img.shape[:2]
print('image size = ',img_shape)

# specify no of bands in the image
n_bands = 33
# 3 dimensional dummy array with zeros
MB_img = np.zeros((img_shape[0],img_shape[1],n_bands))

# stacking up images into the array
for i in range(n_bands):
    MB_img[:,:,i] = imread(str(i+1)+'.png',as_gray=True)  

# Let's take a look at scene
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
print('\n\nDispalying 1st 10th and 30th band of image')
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(img10, cmap=plt.cm.gray)
ax3.imshow(img30, cmap=plt.cm.gray)

ax1.set_title(r'1st band', fontsize=15)
ax2.set_title(r'10th band', fontsize=15)
ax3.set_title(r'30th band', fontsize=15)

fig.tight_layout()
plt.show()

# Spectrum of selected pixels
v = np.zeros(n_bands)
for i in range(n_bands):
    v[i] = MB_img[35,18,i]
   
e = np.zeros(n_bands)
for i in range(n_bands):
    e[i] = MB_img[51,107,i]
    
f = np.zeros(n_bands)
for i in range(n_bands):
    f[i] = MB_img[49,603,i]







    
x = np.arange(33)

plt.figure()
plt.plot(x,v)
plt.plot(x,e)
plt.plot(x,f)
plt.title('Spectrum Plot')
plt.ylabel('Reflectance')
plt.xlabel('band # ')

# Calculating Kmean for clustring data
# Unsupervised classification algorithms divide image pixels into groups based on 
# spectral similarity of the pixels without using any prior knowledge of the spectral classes.

mean1=np.mean(MB_img,axis=0)
MB_img1=MB_img-mean1
# # q=imsave("new.jpg",MB_img)

# mean_row=1/50787*np.sum([X],axis=1)
# ones=np.ones([33,1])
# mean_a=np.dot(mean_row.T,ones.T)
# B=X-mean_a
(m, c) = kmeans(MB_img1, 3, 5)

plt.figure()

for i in range(c.shape[0]):
    plt.plot(c[i])
plt.grid()
plt.title('Spectral Classes from K-Means Clustering')
plt.ylabel('Reflectance')
plt.xlabel('band # ')    


# saving matlab file for index
sio.savemat('idx1.mat', {'m': m})

# Feature Extraction
# Applying PCA  
# pc = principal_components(MB_img)
# pc.cov shows covariance matrix display, lighter values indicate strong positive covariance, 
# darker values indicate strong negative covariance, and grey values indicate covariance near zero.
plt.figure()
pc = principal_components(MB_img)
pc_view = imshow(pc.cov, cmap=plt.cm.gray)
xdata = pc.transform(MB_img)

# Reduces the number of principal components. 
# Eigenvalues will be retained (starting from greatest to smallest) until fraction of total image 
# variance is retained i.e. all major PC's are retained below.
pcdata = pc.reduce(num=3).transform(MB_img)
pc_0999 = pc.reduce(fraction=0.999)

# How many eigenvalues are left?
print(len(pc_0999.eigenvalues))

img_pc = pc_0999.transform(MB_img)
print(img_pc.shape)

plt.figure()
v = imshow(img_pc[:,:,:3])




#sio.savemat('new.mat',q)
# import glob 
# import cv2
# import sys
# while 1 :
#     filename = raw_input("HSI")
#     for img in glob.glob(filename+'/*.*'):
#         try :
#             var_img = cv2.imread(img)
#             cv2.imshow(str(img) , var_img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
        
#         except Exception as e:
#             print (e)
#     user_input = raw_input("do you want to read another folder = ")
#     if user_input == 'no':
#         break

# 81,627,3 for matlab
n_bands = 3
# 3 dimensional dummy array with zeros
MB_img2 = np.zeros((img_shape[0],img_shape[1],n_bands))


sio.savemat('matrix-3_dim.mat', {'MB_img2': MB_img2})



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage
# from skimage.io import imread
# from skimage.io import imshow
# from skimage.transform import radon, rescale


# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5))
# ax1.set_title("Original")
# image =imread('1.png', plugin=None)

# new_img = np.zeros((81,627,2))
# new_img[:,:,0]=image