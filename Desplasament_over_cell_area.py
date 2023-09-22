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

muestra, region = "D", 2

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
n = 0
pre = stack_pre[ n ]
# post, m, XY = correct_driff( stack_post[ n-1 ] , pre, 50, info = True)
post = correct_driff_3D( stack_post, pre, 50)

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

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)

x, y = dominio
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
l = len(Y_work)
# print( ks*(0.5 - th)*ps )

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
d_ra = np.sqrt(  (x_a**2)/( r_a )*dx_a  +  (y_a**2)/( r_a )*dx_a   )

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


#%%

#%% 21/9

dr = 10 # um
ps = pixel_size
dr_px = dr/ps
print(dr_px)

th = 0.1
ks = int(np.round( dr_px/0.4 ))
print(ks)

#%%

m0 = np.copy(mascara)
ms = np.copy(mascara)
for j in range(dr):
    m0 = area_upper(m0, kernel_size = ks//dr, threshold = th)
    ms = ms + m0
    # plt.imshow(msuma)
#%%

plt.imshow( ms )

#%%
corte = ( ms[510, 620:770] )
plt.plot( corte )

#%%

corte_diff = np.diff(corte)
plt.plot( corte_diff )

#%%
puntos = []

for i in range( len(corte_diff) ):
    if corte_diff[i] < -0.5:
        puntos.append( i )

print( np.mean( np.diff(puntos)*ps ) )


#%%

#%%
ker_list = []
incremento = []

for i in range(9, 51, 2):
    m1 = np.zeros([1024]*2)
    m1[:,500:700] = 1

    m2 = area_upper(m1, kernel_size = int(i/0.4), threshold = 0.1)
    msuma = m2+m1
    msuma_diff = np.diff(msuma[512])

    puntos_subida = []
    puntos_bajada = []
    for j in range( len(msuma_diff) ):
        if msuma_diff[j] > 0.5:
            puntos_subida.append( j )
        if msuma_diff[j] < -0.5:
            puntos_bajada.append( j )

    print(i)
    ker_list.append(i)
    incremento.append( ( np.diff(puntos_subida)[0] + np.diff(puntos_bajada)[0])/2  )


#%%

plt.plot(np.array(ker_list), incremento, 'o')
plt.xlabel("Tamaño del kernel [um]")
plt.ylabel("Incremento [um]")
plt.grid(True)

m = np.mean(np.diff(np.array(incremento).flatten())/np.diff(np.array(ker_list) )  )
print(m)
#%%

def recta(x, m, b):
    return m*x + b

popt, pcov = curve_fit(recta, ker_list, np.array(incremento).flatten())
print(popt, np.sqrt(np.diag(pcov)))


#%% 22/9

Y_work, X_work = Y_s, X_s
l = len(Y_work)

dr = 30 # um
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
    defo.append( np.sqrt( np.array(Y_cell)**2 + np.array(X_cell)**2 ) )
                
#%% 
   
plt.grid(True)
sns.violinplot(data=defo, x=None, y=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=500, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None)
plt.ylabel("Deformación [um]")
plt.xlabel("Incremeto [um]")
plt.xticks(np.arange(dr//2),np.arange(dr//2)*2)


#%%
plt.figure()                
plt.plot(defo)

plt.figure()
plt.imshow( ms )

plt.figure()
plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = "Oranges", alpha = 0.4 )
plt.imshow( m0, cmap = "Oranges", alpha = 0.4 )
plt.quiver(x,y,X_s,-Y_s, scale = 100, pivot='tail')
plt.xlim([0, resolution])
plt.ylim([resolution,0])

