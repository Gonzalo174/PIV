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

# Viejos

# file_cell = "gel1_cell1_TFM_47nm_PRE_K.oif"
# file_pre = "gel1_cell1_TFM_47nm_PRE_K.oif"
# file_post = "gel1_cell1_TFM_47nm_Post_K.oif"

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

# file_cell = "B1-R2-10-60X-pw0.5-k0-tra.oif"
# file_pre = "B1-R2-11-60X-pw20-k2-pre.oif"
# file_post = "B1-R2-12-60X-pw20-k2-post.oif"

file_cell = "B1-R3-06-60X-pw0.5-k0-tra.oif"
file_pre = "B1-R3-07-60X-pw20-k2-pre.oif"
file_post = "B1-R3-14-60X-pw20-k2-post.oif"

stack_pre = of.imread( file_pre )[0]
stack_post = of.imread( file_post )[0]
celula = of.imread( file_cell )[1, 1]

mascara1 = iio.imread( "celula_Crimson_R3_cell1.png" )
mascara2 = iio.imread( "celula_Crimson_R3_cell2.png" )
mascara3 = iio.imread( "celula_Crimson_R3_cell3.png" )
mascara4 = iio.imread( "celula_Crimson_R3_cell4.png" )
mascara5 = iio.imread( "celula_Crimson_R3_cell5.png" )

#%% Analize correlation

n = 0
pre1 = stack_pre[ n ]
post0 = centrar_referencia( stack_post[ n ] , pre1, 250)
# post0 = centrar_referencia_3D( stack_post, pre1, 250)


#%% Pre-Post-Trans Plot 

plt.figure()
plt.title('Pre')
plt.imshow( np.flip( pre1, 0 ) , cmap = 'gray', vmin = 80, vmax = 400)

plt.figure()
plt.title('Post')
plt.imshow(  np.flip( post0, 0 ) , cmap = 'gray', vmin = 80, vmax = 400)

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
exploration = 8 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Fit"

Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration, mode = modo)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
suave0 = 3
X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)



#%% Plot

l = len(Y_nmt)
scale0 = 150
wind = vi/( 2**(it-1) )
field_length = int( l*wind )
image_length = len( celula )
d = (image_length - field_length)/2
r_plot = np.arange(l)*wind + wind/2 + d

x,y = np.meshgrid( r_plot , r_plot )

plt.figure()
plt.title("Mapa de deformación - " + modo)

# plt.imshow( celula , cmap = 'gray' , alpha = 0.5)
plt.imshow( 1-mascara1 , cmap = 'Greens', alpha = 0.2 )
plt.imshow( 1-mascara2 , cmap = 'Reds', alpha = 0.2 )
plt.imshow( 1-mascara3 , cmap = 'Blues', alpha = 0.2 )
plt.imshow( 1-mascara4 , cmap = 'Oranges', alpha = 0.2 )
plt.imshow( 1-mascara5 , cmap = 'Purples', alpha = 0.2 )

# plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0)
plt.quiver(x,y,X_s,-Y_s, scale = scale0)

scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + wind  # Starting x-coordinate of the scale bar
start_y = image_length -( d + wind/2 + 40 )# Starting y-coordinate of the scale bar

plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_bar_length, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')

plt.xticks([])
plt.yticks([])
plt.xlim([d,image_length-d])
plt.ylim([image_length-d,d])

plt.show()

#%%
r = np.sqrt(Y_s**2 + X_s**2)*pixel_size
r_mean = np.mean( r.flatten()*pixel_size )
plt.figure()
plt.title("Distribucion NMT suavizado, r_mean: " + str( np.round(r_mean,3)) + ' um'  )
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
# plt.ylim([0,600])
plt.hist(r.flatten(), bins = np.arange(-0.01, np.round( np.max(r)+0.01 ,1 ) , 0.02)  )
plt.show()


#%% Nanospheres intensity histogram
val_pre = pre1.flatten()
val_post = post0.flatten()

plt.figure()
plt.hist(val_pre, bins = np.arange(4000))
plt.title('pre')

plt.figure()
plt.hist(val_post, bins = np.arange(4000))
plt.title('post')


#%% gif

n = 0
# vi = 256
windows = [200, 220, 256, 280, 300]
it = 3
exploration = 5 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

image_filenames = []

for vi in windows:
    pre1 = stack_pre[ n ]
    post0 = centrar_referencia( stack_post[ n ] , pre1, 250)

    Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
    Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
    suave0 = 3
    X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)
    
    l = len(Y_nmt)
    scale0 = 100
    wind = vi/( 2**(it-1) )
    field_length = int( l*wind )
    image_length = len( celula )
    d = (image_length - field_length)/2
    r_plot = np.arange(l)*wind + wind/2 + d
    
    x,y = np.meshgrid( r_plot , r_plot )
    
    plt.figure()
    # plt.title("Mapa de deformación - z = -" + str(n/2 + 3) + ' µm')
    plt.title("Mapa de deformación - w = " + str(int(vi/4)) + ' px')
    
    plt.imshow( 1-mascara1 , cmap = 'Greens', alpha = 0.2 )
    plt.imshow( 1-mascara2 , cmap = 'Reds', alpha = 0.2 )
    plt.imshow( 1-mascara3 , cmap = 'Blues', alpha = 0.2 )
    plt.imshow( 1-mascara4 , cmap = 'Oranges', alpha = 0.2 )
    plt.imshow( 1-mascara5 , cmap = 'Purples', alpha = 0.2 )
    
    # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0)
    plt.quiver(x,y,X_s,-Y_s, scale = scale0)
    
    scale_length = 10  # Length of the scale bar in pixels
    scale_pixels = scale_length/pixel_size
    scale_unit = 'µm'  # Unit of the scale bar
    scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    start_x = 100 #d + wind  # Starting x-coordinate of the scale bar
    start_y = image_length - 100 #-( d + wind/2 + 40 )# Starting y-coordinate of the scale bar
    plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
    plt.text(start_x + scale_bar_length, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')
    plt.xticks([])
    plt.yticks([])
    # plt.xlim([d,image_length-d])
    # plt.ylim([image_length-d,d])
    name =  str(vi) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )


#%% gif

vi = 256
it = 3
exploration = 5 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

image_filenames = []

for vi in windows:
    pre1 = stack_pre[ n ]
    post0 = centrar_referencia( stack_post[ n ] , pre1, 250)

    Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
    Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
    suave0 = 3
    X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)


    r = np.sqrt(Y_s.flatten()**2 + X_s.flatten()**2)*pixel_size
    r_mean = np.mean( r )
    plt.figure()
    plt.title( "w = "  + str(int(vi/4)) + " px, Deformación promedio = " + str( np.round(r_mean,3)) + ' µm'  )
    plt.xlabel('Desplazamiento [µm]')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.hist(r, bins = np.arange(-0.04, 0.6 , 0.08) , density=True )
    plt.plot( [r_mean,r_mean], [0,4], color = "r", linestyle = "dashed", lw = 4 )
    plt.xlim([-0.06,0.66])
    plt.ylim([0,4.1])
    name =  str(vi) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )
    
#%% gif maker

from PIL import Image

# Open the images and convert them to the same format
images = []
for filename in image_filenames:
    image = Image.open(filename)
    # image = image.convert('RGBA')  # Convert to RGBA format (optional, for transparency support)
    images.append(image)

# Save the images as an animated GIF
output_filename = 'res.gif'  # Specify the output filename
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=500, loop=0)














#%% Fuerza en la celula

vi = 256
it = 3
exploration = 5 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
suave0 = 3
X_s,Y_s = suavizar(X_nmt, suave0),suavizar(Y_nmt, suave0)
X_work, Y_work = np.copy(X_s), np.copy(Y_s)
X_work[:3,:4], Y_work[:3,:4] = np.zeros([3,4]), np.zeros([3,4])

l = len(Y_nmt)
scale0 = 50
wind = vi/( 2**(it-1) )
field_length = int( l*wind )
image_length = len( celula )
d = (image_length - field_length)/2
r_plot = np.arange(l)*wind + wind/2 + d
x,y = np.meshgrid( r_plot , r_plot )
    

#%% Desplacement at restriged areas

cell_area = 1 - mascara1 
cell_area_down = np.copy( cell_area )
ps = pixel_size
il = len(cell_area)
ks = 40
th = 0.9
print( ks*(0.5 - th)*ps )

n_r = 10
x_a = np.zeros([n_r*2-1])
y_a = np.zeros([n_r*2-1])
dx_a = np.zeros([n_r*2-1])
dy_a = np.zeros([n_r*2-1])
a = np.zeros([n_r*2-1, il, il])

for n in range(n_r):
    # Para afuera
    if n !=0:
        cell_area = area_upper( cell_area, ks, th)

    Y_cell = []
    X_cell = []

    for j in range(l):
        for i in range(l):
            if cell_area[ int(x[j,i]), int(y[j,i]) ] == 1:
                Y_cell.append(Y_work[j,i])
                X_cell.append(X_work[j,i])

    x_a[n+n_r-1] = np.mean( X_cell )
    y_a[n+n_r-1] = np.mean( Y_cell ) 
    dx_a[n+n_r-1] = np.std( X_cell )
    dy_a[n+n_r-1] = np.std( Y_cell ) 
    a[n+n_r-1] = cell_area
    
    # Para adentro
    cell_area_down = area_upper( cell_area_down, ks, 1-th)
    
    Y_cell = []
    X_cell = []

    for j in range(l):
        for i in range(l):
            if cell_area_down[ int(x[j,i]), int(y[j,i]) ] == 1:
                Y_cell.append(Y_work[j,i])
                X_cell.append(X_work[j,i])
    
    print( n+1 )
    x_a[-n+n_r-2] = np.mean( X_cell )
    y_a[-n+n_r-2] = np.mean( Y_cell ) 
    dx_a[-n+n_r-2] = np.std( X_cell )
    dy_a[-n+n_r-2] = np.std( Y_cell ) 
    a[-n+n_r-2] = cell_area_down
    
r_a = np.sqrt( x_a**2 + y_a**2 )
d_ra = np.sqrt(  (x_a**2)*( r_a )**3  +  (y_a**2)*( r_a )**3   )

iio.imwrite("a.tiff",a)

#%%
r_plot = -(np.arange(len(r_a)) - n_r )
dr_plot = np.ones(len(r_a))*ps*3

plt.title("Deformación resultante")
plt.errorbar(x = -r_plot, y = r_a, yerr=d_ra/2, xerr=dr_plot, fmt = '.')
plt.grid(True)
plt.xlabel("dr [µm]")
plt.ylabel("Deformación resultante [µm]")

#%%
CM = center_of_mass(mascara)
mapita = np.sum( a, 0 )
    
plt.imshow( mapita , cmap = "plasma")
plt.plot( [CM[1]],[CM[0]] , 'o')
plt.xlim([0,1000])
plt.ylim([1000,0])

#%% gif

CM = center_of_mass(mascara)
scale1 = 3
scale0 = 30
image_filenames = []

for i in range(len(a)):
    alpha = 0.2
    # if i - n_r  == 0:
    #     alpha = 0.5
    plt.figure()
    plt.title("Deformación resultante - dr = " + str( -(i - n_r ) ) + ' µm')
    plt.imshow( a[i-1] , cmap = 'Blues', alpha=alpha )
    plt.imshow( 1-mascara1 , cmap = 'Blues', alpha=0.3)
    plt.quiver([CM[1]],[CM[0]],x_a[-i-1],-y_a[-i-1], scale = scale1)
    plt.quiver(x,y,X_s,-Y_s, scale = scale0)
    
    scale_length = 10  # Length of the scale bar in pixels
    scale_pixels = scale_length/pixel_size
    scale_unit = 'µm'  # Unit of the scale bar
    scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    start_x = 800 # Starting x-coordinate of the scale bar
    start_y = 950 # Starting y-coordinate of the scale bar
    plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
    for i0 in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i0 - 10, start_y + i0 - 10], color='black', linewidth = 1)
    plt.text(start_x + scale_bar_length, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,1000])
    plt.ylim([1000,0])
    # plt.xlim([d,image_length-d])
    # plt.ylim([image_length-d,d])
    name =  str( -(i - n_r ) ) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )













#%%
plt.figure()
plt.title("Mapa de deformación")

# plt.imshow( celula , cmap = 'gray' , alpha = 0.5)
plt.imshow( mascara , cmap = 'Greens', alpha = 0.2 )

plt.quiver(x,y,X_work,-Y_work, scale = scale0)

scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + wind  # Starting x-coordinate of the scale bar
start_y = image_length -( d + wind/2 + 40 )# Starting y-coordinate of the scale bar

plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_bar_length, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')

plt.xticks([])
plt.yticks([])
# plt.xlim([d,image_length-d])
# plt.ylim([image_length-d,d])

plt.xlim([d,1000])
plt.ylim([1000,d])
plt.show()






#%%



plt.figure()
plt.title("Mapa de deformación")

# plt.imshow( celula , cmap = 'gray' , alpha = 0.5)
plt.imshow( 1-mascara1 , cmap = 'Greens', alpha = 0.2 )
plt.imshow( 1-mascara2 , cmap = 'Reds', alpha = 0.2 )
plt.imshow( 1-mascara3 , cmap = 'Blues', alpha = 0.2 )
plt.imshow( 1-mascara4 , cmap = 'Oranges', alpha = 0.2 )
# plt.imshow( 1-mascara5 , cmap = 'Purples', alpha = 0.2 )

# plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0)
plt.quiver(x,y,X_s,-Y_s, scale = scale0)

scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + wind  # Starting x-coordinate of the scale bar
start_y = image_length -( d + wind/2 + 40 )# Starting y-coordinate of the scale bar

plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_bar_length, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')

plt.xticks([])
plt.yticks([])
plt.xlim([d,image_length-d])
plt.ylim([image_length-d,d])

plt.show()


#%%










































