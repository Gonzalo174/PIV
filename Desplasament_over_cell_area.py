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
post, m, XY = correct_driff( stack_post[ n-1 ] , pre, 50, info = True)
# post = correct_driff_3D( stack_post, pre, 50)

print(XY)
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
bordes_extra = 7 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Smooth3"
mapas = False
suave0 = 3

Y, X = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)


#%%
r = np.sqrt(Y_s**2 + X_s**2)*pixel_size
r_mean = np.mean( r.flatten()*pixel_size )
plt.figure()
plt.title("Distribucion NMT suavizado, r_mean: " + str( np.round(r_mean,3)) + ' um'  )
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
# plt.ylim([0,600])
plt.hist(r.flatten(), bins = np.arange(-0.01, np.round( np.max(r)+0.01 ,1 ) , 0.03)  )
plt.show()



#%% Desplacement at restricted areas

Y_work, X_work = np.copy( Y_s ), np.copy( X_s )

cell_area = 1 - mascara
cell_area_down = np.copy( cell_area )
ps = pixel_size
il = len(cell_area)
ks = 40
th = 0.9
print( ks*(0.5 - th)*ps )

N = 10
x_a = np.zeros([N*2-1])
y_a = np.zeros([N*2-1])
dx_a = np.zeros([N*2-1])
dy_a = np.zeros([N*2-1])
a = np.zeros([N*2-1, il, il])

for n in range(N):
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

    x_a[n+N-1] = np.mean( X_cell )
    y_a[n+N-1] = np.mean( Y_cell ) 
    dx_a[n+N-1] = np.std( X_cell )
    dy_a[n+N-1] = np.std( Y_cell ) 
    a[n+N-1] = cell_area
    
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
    x_a[-n+N-2] = np.mean( X_cell )
    y_a[-n+N-2] = np.mean( Y_cell ) 
    dx_a[-n+N-2] = np.std( X_cell )
    dy_a[-n+N-2] = np.std( Y_cell ) 
    a[-n+N-2] = cell_area_down
    
r_a = np.sqrt( x_a**2 + y_a**2 )
d_ra = np.sqrt(  (x_a**2)*( r_a )**3  +  (y_a**2)*( r_a )**3   )

# iio.imwrite("a.tiff",a)

#%%
r_plot = -(np.arange(len(r_a)) - N )
dr_plot = np.ones(len(r_a))*ps*3

plt.title("Deformación resultante")
plt.errorbar(x = -r_plot, y = r_a, yerr=d_ra/2, xerr=dr_plot, fmt = '.')
plt.grid(True)
plt.xlabel("dr [µm]")
plt.ylabel("Deformación resultante [µm]")

#%%

for img in a:
    plt.imshow(img, cmap = 'Reds', alpha=0.1)



#%%

Y_work, X_work = np.copy( Y_s ), np.copy( X_s )

cell_area = 1 - mascara
cell_area_down = np.copy( cell_area )
ps = pixel_size
il = len(cell_area)
ks = 40
th = 0.9
print( ks*(0.5 - th)*ps )

N = 10
x_a = np.zeros([N*2-1])
y_a = np.zeros([N*2-1])
dx_a = np.zeros([N*2-1])
dy_a = np.zeros([N*2-1])
r_a = np.zeros([N*2-1])
a = np.zeros([N*2-1, il, il])

for n in range(N):
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

    x_a[n+N-1] = np.mean( X_cell )
    y_a[n+N-1] = np.mean( Y_cell ) 
    r_a[n+N-1] = np.mean( np.sqrt( np.array( Y_cell )**2 + np.array( X_cell )**2 ) ) 
    dx_a[n+N-1] = np.std( X_cell )
    dy_a[n+N-1] = np.std( Y_cell ) 
    a[n+N-1] = cell_area
    
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
    x_a[-n+N-2] = np.mean( X_cell )
    y_a[-n+N-2] = np.mean( Y_cell )
    r_a[-n+N-2] = np.mean( np.sqrt( np.array( Y_cell )**2 + np.array( X_cell )**2 ) ) 
    dx_a[-n+N-2] = np.std( X_cell )
    dy_a[-n+N-2] = np.std( Y_cell ) 
    a[-n+N-2] = cell_area_down
    
r_a_vector = np.sqrt( x_a**2 + y_a**2 )
d_ra = np.sqrt(  (x_a**2)*( r_a )**3  +  (y_a**2)*( r_a )**3   )

# iio.imwrite("a.tiff",a)

#%%
r_plot = -(np.arange(len(r_a)) - N )
# dr_plot = np.ones(len(r_a))*ps*3

plt.title("Deformación resultante")
# plt.errorbar(x = -r_plot, y = r_a, yerr=d_ra/2, xerr=dr_plot, fmt = '.')
plt.plot(r_plot, r_a, ".")
plt.grid(True)
plt.xlabel("dr [µm]")
plt.ylabel("Deformación resultante [µm]")
