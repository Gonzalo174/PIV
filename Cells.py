# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:29:00 2023

@author: gonza
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os
from skimage import filters


#%%
plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 15

#%%

# Specify the folder path
folder_path = r'C:\Users\gonza\1\Tesis\2023\23.05.18 - Celulas en geles OA Orange'  # Replace with the path to your folder
# folder_path = r'C:\Users\gonza\1\Tesis\2022\practica-PIV\gel8'

# Get the list of file names in the folder
file_names = os.listdir(folder_path)

# Print the file names
for file_name in file_names:
    print(file_name)


#%% Import files and set metadata
# field = 75.2
field = 105.6
resolution = 1600
pixel_size = field/resolution

# Viejos

# file_cell = "gel1_cell1_TFM_47nm_PRE_K.oif"
# file_pre = "gel1_cell1_TFM_47nm_PRE_K.oif"
# file_post = "gel1_cell1_TFM_47nm_Post_K.oif"

# Orange 18/5

# file_cell = "09-BOB-R2-60X-pw0.2-k0-zoomX2-tra.oif"
# file_pre = "10-BOB-R2-60X-pw0.2-k2-zoomX2-pre.oif"
# file_post = "13-BOB-R2-60X-pw0.2-k2-zoomX2-post.oif"

# file_cell = "01-BOA-R1-60X-pw0.1-k0-tra.oif"
# file_pre = "02-BOA-R1-60X-pw0.2-k0-pre.oif"
# file_post = "06-BOA-R1-60X-pw0.2-k2-post.oif"

# file_cell = "03-BOA-R2-60X-pw0.2-k0-tra.oif"
# file_pre = "04-BOA-R2-60X-pw0.2-k2-pre.oif"
# file_post = "05-BOA-R2-60X-pw0.2-k2-post.oif"

# Crimson 11/5

# file_cell = "B1-R1-08-60X-pw0.5-k0-tra.oif"
# file_pre = "B1-R1-09-60X-pw20-k2-pre.oif"
# file_post = "B1-R1-13-60X-pw20-k2-post.oif"

file_cell = "B1-R2-10-60X-pw0.5-k0-tra.oif"
file_pre = "B1-R2-11-60X-pw20-k2-pre.oif"
file_post = "B1-R2-12-60X-pw20-k2-post.oif"

# file_cell = "B1-R3-06-60X-pw0.5-k0-tra.oif"
# file_pre = "B1-R3-07-60X-pw20-k2-pre.oif"
# file_post = "B1-R3-14-60X-pw20-k2-post.oif"

# stack_pre = of.imread( file_pre )[0]
# stack_post = of.imread( file_post )[0]

#%% Plot 

celula = of.imread( file_cell )[1, 1]
plt.figure()
# plt.title('Trans')
plt.imshow( celula  )
plt.axis('off')

#%%

celula2 = suavizar(celula,2)
plt.hist( celula2.flatten(), bins = np.arange(4096) )
plt.xlim(800,1700)

#%%

# umbral = filters.threshold_otsu(celula2, nbins = np.arange(4096))
celula_bin = np.zeros( celula.shape )
celula_bin[celula2>1250] = 1
plt.imshow( celula_bin  )
plt.axis('off')

#%%
# kernel del filtro
s = [[1, 2, 1],  
     [0, 0, 0], 
     [-1, -2, -1]]

HR = signal.convolve2d(celula, s)
VR = signal.convolve2d(celula, np.transpose(s))
img_bordes = (HR**2 + VR**2)**0.5

plt.figure()
plt.imshow( img_bordes )
plt.title('Filtro de detección de bordes')
#%%




























