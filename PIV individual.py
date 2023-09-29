# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:34:21 2023

@author: Usuario
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16

c0 = [(0, 0, 0), (0, 0, 0)]
cm0 = ListedColormap(c0)
c3 = [(1, 1, 1), (1, 1, 0)]
cm3 = ListedColormap(c3)

c1 = []
c2 = []
for i in range(1000):
    c1.append( (i/999,0,0) )
    c2.append( (0,i/999,0) )

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)

#%% Import

CS =      [ (11,2), (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
pre_CS =  [      6,      4,      4,       4,      8,      4,     3,     9 ]
post_CS = [      5,      3,      4,       3,      4,      4,     3,     8 ]

SS =      [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (3,3) ]
pre_SS =  [     4,     4,     6,     5,     6,     5,     4,     3,     6 ]
post_SS = [     4  ,   4,     4,     4,     5,     4,     5,     4,     6 ]

Cualitativas = [ (8,1), (8,5), (7,2), (6,1), (5,3), (5,5), (3,5) ]

# path = r"C:\Users\gonza\1\Tesis\2023\\"
path = r"D:\Gonzalo\\"
carpetas = ["23.08.17 - gon MCF7 1", "23.08.18 - gon MCF7 2", "23.08.18 - gon MCF7 3", "23.08.24 - gon MCF7 4", "23.08.24 - gon MCF7 5", "23.08.25 - gon MCF7 6", "23.08.25 - gon MCF7 7", "23.08.31 - gon MCF7 8 - B30", "23.08.31 - gon MCF7 9 - A30", "23.09.01 - gon MCF7 10 - C30", "23.09.01 - gon MCF7 11 - D30"]
muestras = [ "C16", "B16", "A16", "A23", "B23", "D23", "C23", "B30", "A30", "C30", "D30" ]


#%%
i = -1
tup = CS[i]


name =  muestras[tup[0]-1] + 'R' + str(tup[1])
print(name)

full_path = path + carpetas[int(tup[0]-1)] + '\Data.csv'
region = int( tup[1]  )
metadata = pd.read_csv( full_path, delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == region ]

field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

stack_pre = of.imread( path + carpetas[int(tup[0]-1)] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( path + carpetas[int(tup[0])-1] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
mascara = 1- iio.imread(path + r"PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + ".png")

pre = stack_pre[ 10 ]#pre_CS[i] ]
post, YX = correct_driff_3D( stack_post , pre, 50, info = True)
   

vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
# modo = "No control"
modo = "Smooth3"

mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
R_s = np.sqrt( X_s**2 + Y_s**2 )
x, y = dominio


inf = 120
a = np.mean(post)/np.mean(pre)
pre_plot = np.copy( (pre+5)*a - inf )
post_plot = np.copy(post - inf )
pre_plot[ pre < 0 ] = 0
pre_plot[ post < 0 ] = 0

scale0 = 100
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

wind = vi/( 2**(it-1) )
d = int( ( resolution - len(Y_nmt)*wind )/2   )

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + 80  # Starting x-coordinate of the scale bar
start_y = resolution -( 2*wind )# Starting y-coordinate of the scale bar

plt.figure(figsize=(20,20), tight_layout=True)

# Pre
plt.subplot(2,2,1)

plt.imshow( pre, cmap = 'gray', vmin = 80, vmax = 700)
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")

# Post
plt.subplot(2,2,2)

plt.imshow( post, cmap = 'gray', vmin = 80*a, vmax = 700*a)
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

# Esferas
plt.subplot(2,2,3)
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)

# Deformacion
plt.subplot(2,2,4)

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = cm3, alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.show()




