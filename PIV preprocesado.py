# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:50:42 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16

#%%
region = 1
metadata = pd.read_csv('Data.csv', delimiter = ',', usecols=np.arange(3,15,1))
otradata = pd.read_csv('Data.csv', delimiter = ',', usecols=np.arange(0,2,1), index_col=0, header=0, names = ['1','2'])
metadata_region = metadata.loc[ metadata["Región"] == region ]

stack_pre = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
celula = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'TRANS' ]["Archivo"].values[0]+".oif" )[1]
celula_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1][0]


field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

pre = stack_pre[ 6 ]
post, ZYX = correct_driff_3D( stack_post, pre, 50, info = True)

# post0 = stack_post[ 3 ]
# post, m, YX = correct_driff( post0 , pre, 50, info = True)


print(metadata_region["Archivo"].values)
print(n_pre, n_post  )
print(ZYX)

a = np.mean(post)/np.mean(pre)

plt.figure()
plt.title('Pre')
plt.imshow( pre, cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post, cmap = 'gray', vmin = 80*a, vmax = 700*a  )

plt.figure()
plt.title('Trans')
plt.imshow( celula , cmap = 'gray' )

plt.figure()
plt.title('Trans')
plt.imshow( celula_post , cmap = 'gray' )



# %% PIV + NMT + Suavizado

vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Smooth3"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post_rotada, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)

x, y = dominio


name = otradata.values[32][0] + "_r" + str(region) + "_z" + str( metadata_region["Zoom"].values[0] )

inf = 120
a = np.mean(post)/np.mean(pre)
pre_plot = np.copy( (pre+5)*a - inf )
post_plot = np.copy(post_rotada - inf )
pre_plot[ pre < 0 ] = 0
pre_plot[ post < 0 ] = 0

c0 = [(0, 0, 0), (0, 0, 0)]
cm0 = ListedColormap(c0)

c1 = []
c2 = []
for i in range(1000):
    c1.append((i/999,0,0))
    c2.append((0,i/999,0))

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)

scale0 = 100
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

wind = vi/( 2**(it-1) )
d = int( ( resolution - len(Y_nmt)*wind )/2   )

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + 0  # Starting x-coordinate of the scale bar
start_y = resolution -( 2*wind )# Starting y-coordinate of the scale bar



plt.figure(figsize=(20,20), tight_layout=True)

plt.subplot(2,2,1)

plt.imshow( celula , cmap = 'gray' )
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")


plt.subplot(2,2,2)

plt.imshow( celula_post , cmap = 'gray' )
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)



plt.subplot(2,2,3)
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)


plt.subplot(2,2,4)

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.savefig(name + '_figura.png')
plt.show()


c_pre_a = np.reshape( celula, (1, 1, *pre.shape)  )
c_post_a = np.reshape( celula_post, (1, 1, *pre.shape)  )

pre_a = np.reshape( pre, (1, 1, *pre.shape)  )
post_a = np.reshape( post, (1, 1, *pre.shape)  ).astype(dtype=np.uint16)


archivo1 = np.concatenate( (c_pre_a,c_post_a), axis = 1 )
archivo2 = np.concatenate( (pre_a,post_a) , axis = 1 )

archivo0 = np.concatenate( (archivo1,archivo2), axis = 0 )


iio.imwrite( name+".tiff" , archivo0, extension = '.tiff' )





