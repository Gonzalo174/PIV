# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:56:55 2023

@author: Gon
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 15

#%% Import files and set metadata
field = 105.6
resolution = 1600
pixel_size = field/resolution

# Orange 18/5

# file_cell = "01-BOA-R1-60X-pw0.1-k0-tra.oif"
# file_pre = "02-BOA-R1-60X-pw0.2-k0-pre.oif"
# file_post = "06-BOA-R1-60X-pw0.2-k2-post.oif"

# file_cell = "03-BOA-R2-60X-pw0.2-k0-tra.oif"
# file_pre = "04-BOA-R2-60X-pw0.2-k2-pre.oif"
# file_post = "05-BOA-R2-60X-pw0.2-k2-post.oif"

# file_cell = "09-BOB-R2-60X-pw0.2-k0-zoomX2-tra.oif"
# file_pre = "10-BOB-R2-60X-pw0.2-k2-zoomX2-pre.oif"
# file_post = "13-BOB-R2-60X-pw0.2-k2-zoomX2-post.oif"

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

stack_pre = of.imread( file_pre )[0]
stack_post = of.imread( file_post )[0]
celula = of.imread( file_cell )[1, 1]

#%% Analize correlation

pre1 = stack_pre[ 1 ]
post0 = centrar_referencia( stack_post[ 1 ] , pre1, 250)
# post0 = centrar_referencia_3D( stack_post, pre1, 250)

#%%
mascara1 = iio.imread( "celula_Orange_R2_cell1.png" )
mascara2 = iio.imread( "celula_Orange_R3_cell2.png" )
mascara3 = iio.imread( "celula_Crimson_R3_cell3.png" )
mascara4 = iio.imread( "celula_Crimson_R3_cell4.png" )
# mascara5 = iio.imread( "celula_Crimson_R3_cell5.png" )

#%% Plot 

plt.figure()
plt.title('Pre')
plt.imshow( np.flip( pre1, 0 ) , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow(  np.flip( post0, 0 ) , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Trans')
plt.imshow( np.flip( celula , 0 ) , cmap = 'gray' )
# plt.imshow( np.flip( 1-mascara1  , 0 ) , cmap = 'Greens', alpha = 0.2 )
# plt.imshow( np.flip( 1-mascara2  , 0 ) , cmap = 'Reds', alpha = 0.2 )
# plt.imshow( np.flip( 1-mascara3  , 0 ) , cmap = 'Blues', alpha = 0.2 )


# plt.imshow( np.flip( b , 0 ), cmap = "gray")


#%% Reconstruyo con PIV y filtro los datos con, Normalized Median Test (NMT)
vi = 128
it = 3
exploration = 5 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
suave0 = 3
X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)


#%%
r = np.sqrt(Y_s**2 + X_s**2)
r_mean = np.mean( r.flatten()*pixel_size )
plt.figure()
plt.title("Distribucion NMT suavizado, r_mean: " + str( np.round(r_mean,3)) + ' um'  )
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
# plt.ylim([0,600])
plt.hist(r.flatten()*pixel_size, bins = np.arange(-0.01, np.round( (exploration+1)*pixel_size,1 ) , 0.02)  )
plt.show()




#%%

l = len(Y_nmt)
scale0 = 100
field_length = int( l*vi/it )
image_length = len( celula )
d = (image_length - field_length)
r_plot = np.arange(l)*vi/it + d

x,y = np.meshgrid( r_plot , r_plot )

# plt.figure()
# plt.title('Resultado NMT')
# plt.yticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
# plt.xticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# # plt.imshow(40 - b, cmap = "gray")
# plt.quiver(x,y,X_nmt,Y_nmt, scale = scale0)

plt.figure()
plt.title("Mapa de deformación")


# plt.imshow( celula , cmap = 'gray' , alpha = 0.5)
# plt.imshow( 1-mascara1 , cmap = 'Greens', alpha = 0.2 )
# plt.imshow( 1-mascara2 , cmap = 'Reds', alpha = 0.2 )
# plt.imshow( 1-mascara3 , cmap = 'Blues', alpha = 0.2 )
# plt.imshow( 1-mascara4 , cmap = 'Oranges', alpha = 0.2 )
# plt.imshow( 1-mascara5 , cmap = 'Purples', alpha = 0.2 )
plt.quiver(x,y,X_s,Y_s, scale = scale0)

#%%
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = 1335  # Starting x-coordinate of the scale bar
start_y = 155  # Starting y-coordinate of the scale bar

plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_bar_length - 15, start_y - 30, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')

# Remove the axis ticks and labels
plt.xticks([])
plt.yticks([])

# Display the image with scale bar
plt.show()

# plt.yticks( np.arange(6)*20/pixel_size , np.arange(6)*20 )
# plt.xticks( np.arange(6)*20/pixel_size , np.arange(6)*20 )
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")


#%%

scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = 1335  # Starting x-coordinate of the scale bar
start_y = 155  # Starting y-coordinate of the scale bar

plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_bar_length - 15, start_y - 30, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')

# Remove the axis ticks and labels
plt.xticks([])
plt.yticks([])

# Display the image with scale bar
plt.show()

#%%






r = np.sqrt(Y_nmt**2 + X_nmt**2)
plt.figure()
plt.title("Distribucion NMT")
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
# plt.ylim([0,600])
plt.hist(r.flatten()*pixel_size, bins = np.arange(-0.01, np.round( (exploration+1)*pixel_size, 1 ) , 0.02)  )
print(np.mean( r.flatten()*pixel_size ))
#%%

r = np.sqrt(Y_s**2 + X_s**2)
r_mean = np.mean( r.flatten()*pixel_size )
plt.figure()
plt.title("Distribucion NMT suavizado, r_mean: " + str( np.round(r_mean,3)) + ' um'  )
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
# plt.ylim([0,600])
plt.hist(r.flatten()*pixel_size, bins = np.arange(-0.01, np.round( (exploration+1)*pixel_size,1 ) , 0.02)  )
plt.show()


#%%

iio.imwrite('post0.tiff', post0)
iio.imwrite('post1.tiff', post1)

#%%
val_pre = pre1.flatten()
val_post = post0.flatten()

plt.figure()
plt.hist(val_pre, bins = np.arange(4000))
plt.title('pre')

plt.figure()
plt.hist(val_post, bins = np.arange(4000))
plt.title('post')

