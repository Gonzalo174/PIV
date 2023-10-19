# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:23:33 2023

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
plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 16


#%% Import
# path = r"C:\Users\gonza\1\Tesis\2023\\"
path = r"D:\Gonzalo\\"
carpetas = ["23.10.05 - gon MCF10 1 - A04", "23.10.05 - gon MCF10 2 - D04", "23.10.05 - gon MCF10 3 - E04", "23.10.06 - gon MCF10 4 - C04", "23.10.19 - gon MCF10 6 - G18" ]
# muestras = [ "C16", "B16", "A16", "A23", "B23", "D23", "C23", "B30", "A30", "C30", "D30" ]

#%% D04_R6
r = 22
distribucion = [ 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5 ]
full_path1 = path + carpetas[distribucion[r]-1]

name = carpetas[distribucion[r]-1][-3:] + "_R" + str(int(r))

metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == r ]

celula = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,0]
plt.figure()
plt.title( name )
plt.imshow(celula, cmap = "gray")



field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
# mascara = 1- iio.imread(path + r"PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + ".png")
# mascara = mascara3

#%%

def normalizar(array):
    return (array - min(array))/max(array - min(array))

err_pre = []
err_post = []


for i in range(len(stack_post)):
    err_post.append( np.std( stack_post[i] ) )
    
for i in range(len(stack_pre)):
    err_pre.append( np.std( stack_pre[i] )   )
    
plt.plot( normalizar(err_pre), label = "PRE" )
plt.plot( normalizar(err_post), label = "POST" )
plt.grid(True)
plt.legend()
plt.title(name)

#%%

n0, dn0 = 5, 2
pre = stack_pre[ n0 ]
post = correct_driff( stack_post[ n0 + dn0 ], pre, 50 )
# post, YX = correct_driff_3D( stack_post[n0-2:] , pre, 50, info = True)
   
a = np.mean(post)/np.mean(pre)

plt.figure()
plt.title('Pre' + str(n0))
plt.imshow( pre, cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post' + str(n0 + dn0))
plt.imshow( post, cmap = 'gray', vmin = 80*a, vmax = 700*a  )


#%%
vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
# modo = "No control"
modo = "Smooth3"

mapas = False
suave0 = 3

# pre[950:,900:] = np.mean( pre[:950,:900] )
# post[950:,900:] = np.mean( post[:950,:900] )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
R_s = np.sqrt( X_s**2 + Y_s**2 )
x, y = np.meshgrid( np.arange(len(X_s)), np.arange(len(Y_s)) )

scale1 = 500

plt.figure()
R_s = np.sqrt( Y_s**2 + X_s**2 )*pixel_size
plt.imshow( R_s )
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.quiver(x,y,X_s/R_s,-Y_s/R_s, scale = scale1, pivot='middle')



#%%
vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
# modo = "No control"
modo = "Smooth3"

mapas = False
suave0 = 3

# pre[950:,900:] = np.mean( pre[:950,:900] )
# post[950:,900:] = np.mean( post[:950,:900] )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
R_s = np.sqrt( X_s**2 + Y_s**2 )
x, y = dominio


plt.figure()
R_s = np.sqrt( Y_s**2 + X_s**2 )*pixel_size
plt.imshow( R_s )
plt.colorbar()
plt.xticks([])
plt.yticks([])

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

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
# plt.imshow( mascara, cmap = cm3, alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.show()



#%%

Y_work, X_work = Y_s, X_s
l = len(Y_work)

dr = 10 # um
ps = pixel_size
ks = int(np.round( dr/ps/0.4 ))

m0 = np.copy(mascara)
ms = np.copy(mascara)
defo = []

for j in range(0,dr,2):
    print(j)
    m0 = area_upper(m0, kernel_size = ks//dr, threshold = 0.1)
    ms = ms + m0
    
Y_cell = []
X_cell = []

for j in range(l):
    for i in range(l):
        if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(m0[ int(x[j,i]), int(y[j,i]) ]) == 1:
            Y_cell.append(Y_work[j,i])
            X_cell.append(X_work[j,i])

    # defo.append( np.sum( np.sqrt( np.array(Y_cell)**2 + np.array(X_cell)**2 ) ) )
    defo.append( np.sqrt( np.array(Y_cell)**2 + np.array(X_cell)**2 )*ps )
                
plt.hist(defo[-1])
#%% 
   
R_s = np.sqrt( Y_s**2 + X_s**2 )*ps
plt.imshow( R_s )
plt.colorbar()
plt.xticks([])
plt.yticks([])


defo = []

for car in carpetas:
    print(car)
    full_path = path + car
    metadata = pd.read_csv( full_path + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    regiones = np.unique( metadata["Región"].values )[:-1]
    for r in regiones:
        print(r)
        metadata_region = metadata.loc[ metadata["Región"] == r ]

        field = metadata_region["Campo"].values[0]
        resolution = metadata_region["Tamano imagen"].values[0]
        pixel_size = field/resolution

        stack_pre = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
        stack_post = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]


        pre = stack_pre[ 5 ]#pre_CS[i] ]
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
        R_s = np.sqrt( X_s**2 + Y_s**2 )*ps
        x, y = dominio

        defo.append(  R_s.flatten() )


#%%
plt.grid(True)
sns.violinplot(data=defo, x=None, y=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=500, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None)
plt.ylabel("Deformación [um]")
plt.xlabel("Célula")
plt.xticks( np.arange(20)-1, np.arange(20) )

















#%%
# kernel del filtro
s = [[1, 2, 1],  
      [0, 0, 0], 
      [-1, -2, -1]]


HR = signal.convolve2d(celula, s)
VR = signal.convolve2d(celula, np.transpose(s))
celula_bordes = (HR**2 + VR**2)**0.5
  
plt.figure()
plt.imshow( celula_bordes )
plt.title('Filtro de detección de bordes')


#%%

mascara0 = np.zeros( celula_bordes.shape )
mascara0[celula_bordes > 90] = 1
# mascara0[:,:200] = 0

cm = center_of_mass( mascara0 )

for j in range(1024):
    for i in range(1024):
        if np.sum( ( np.array(  cm  ) - np.array( [j,i] ) )**2 ) > 375**2:
            mascara0[j,i] = 0


cm = center_of_mass( mascara0 )

for j in range(1024):
    for i in range(1024):
        if np.sum( ( np.array(  cm  ) - np.array( [j,i] ) )**2 ) > 350**2:
            mascara0[j,i] = 0

plt.imshow( mascara0 )
plt.plot( [cm[1]], [cm[0]], "o", c = "r"  )

#%%

mascara2 = smooth(mascara0, 11)

mascara3 = np.zeros( mascara2.shape )
mascara3[mascara2 > 0.3] = 1
plt.imshow( mascara3 )


#%%


s1 = [[1, 1, 1, 1, 1],  
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1]]

mascara1 = np.zeros([1024]*2)

for j in range(1024-4):
    for i in range(1024-4):
        suma = np.sum( s1*mascara0[j:j+5,i:i+5] )
        if suma > 15:
            mascara1[j+2,i+2] = 1



#%%

for car in carpetas:
    full_path = path + car
    metadata = pd.read_csv( full_path + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    regiones = np.unique( metadata["Región"].values )[:-1]
    print(regiones)
    for region in regiones:
        metadata_region = metadata.loc[ metadata["Región"] == region ]
        celula = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,0]




#%% Guardar las imagenes de las celulas para hacer las mascaras

for car in carpetas:
    full_path = path + car
    metadata = pd.read_csv( full_path + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    regiones = np.unique( metadata["Región"].values )[:-1]
    print(regiones)
    for region in regiones:
        metadata_region = metadata.loc[ metadata["Región"] == region ]
        celula = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,0]
        # print( celula.shape )
    
        name = car[-3:] + "_R" + str(int(region)) + ".tiff"
        
        plt.figure()
        plt.title( name[:-5] )
        plt.imshow(celula, cmap = "gray")
        
        # plt.imsave( name, celula, cmap = "gray" )
        iio.imwrite(name, celula)
    
    # print(regiones[:-1])











