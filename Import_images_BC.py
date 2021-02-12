# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:40:42 2021

@author: PCUser
"""

# import
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
#import pickle
#import time
#import random


#from osgeo import gdal#, ogr
from PIL import Image

# %%
#sys.path.append("/scratch/bob/malin_hj/decImages") # Path to images

#import carlsonECG
#ecgs = carlsonECG.load_all_ecgs('/air-data/andersb/data/Expect-Lund-2019-09-23/JonasCarlssonEKG/Expect_Lund_MatFiles/*.mat', 6000)

#dataset = gdal.Open('cd Documents)

#print(dataset.GetMetadata())

im = Image.open('C:/Users/Malin/Documents/LTH/file_example_TIFF_1MB.tiff')
im.show()

# %% 
import matplotlib.pyplot as plt
I = plt.imread('C:/Users/Malin/Documents/LTH/file_example_TIFF_1MB.tiff')
plt.show(I) #works on my computer

# %%
import matplotlib.image as mpimg
import numpy as np
img = mpimg.imread('C:/Users/Malin/Documents/LTH/file_example_TIFF_1MB.tiff')
imgplot = plt.imshow(img)
print(np.max(img))
print(np.min(img))

imgred = img/np.max(img)
print(imgred) #shows the pixel values on Bob