# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:35:01 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 16

#%% Import files and set metadata

muestra, region = "D", 1

metadata = pd.read_csv('data'+muestra+'.csv', delimiter = ',', usecols=np.arange(3,17,1))
metadata_region = metadata.loc[ metadata["Region"] == region ]


field = metadata_region.loc[ metadata_region["Tipo"] == 'pre' ]["Campo"].values[0]
resolution = metadata_region.loc[ metadata_region["Tipo"] == 'pre' ]["Tamano imagen"].values[0]
pixel_size = field/resolution

stack_pre = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'pre' ]["Archivo"].values[0] )[0]
stack_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'post' ]["Archivo"].values[0] )[0]
celula = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'trans pre' ]["Archivo"].values[0] )[1]
celula_redonda = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'trans post' ]["Archivo"].values[0] )[1]
mascara = iio.imread( "mascara" + muestra + str(region) + ".png" )

print(metadata_region["Archivo"].values)

#%% Analize correlation
n = 2
pre = stack_pre[ n ]
post, m, YX = correct_driff( stack_post[ n ] , pre, 50, info = True)
# post = correct_driff_3D( stack_post, pre, 50)

print(YX)
#%% Pre-Post-Trans Plot 

plt.figure()
plt.title('Pre')
plt.imshow( pre, cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow(  post , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Trans')
plt.imshow( celula, cmap = 'gray' )

#%% Reconstruyo con PIV y filtro los datos con, Normalized Median Test (NMT)
vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Smooth3"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)

x, y = dominio
#%% Plot

graficar = 1 
# 0:NMT - 1:Suave
scale0 = 100
alfa = 0.2

plt.figure()
plt.title("Mapa de deformación - " + modo)
plt.imshow( 1-mascara , cmap = 'Reds', alpha = alfa, vmax = 0.1 )

if graficar == 0:
    plt.quiver(x,y,X_nmt,-Y_nmt, np.sqrt( X_nmt**2 + Y_nmt**2 ), scale = scale0, cmap='gist_heat', pivot='tail')
elif graficar == 1:
    plt.quiver(x,y,X_s,-Y_s, np.sqrt( X_s**2 + Y_s**2 ), scale = scale0, cmap='gist_heat', pivot='tail')
    
# plt.colorbar()    
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + wind  # Starting x-coordinate of the scale bar
start_y = image_length -( d + wind )# Starting y-coordinate of the scale bar

plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center')

plt.xticks([])
plt.yticks([])
plt.xlim([d,image_length-d])
plt.ylim([image_length-d,d])

plt.show()






















