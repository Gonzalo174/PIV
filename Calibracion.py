# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:56:20 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
from scipy.optimize import curve_fit
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16

#%% 1 div = 10 um

archivos0 = os.listdir()
archivos = [ ar for ar in archivos0 if ar[-5:] != "files" ]

#%%

plt.imshow( of.imread( archivos[1] )[1], cmap = 'gray' )

#%%

zoom = [ float( ar[4:-4] ) for ar in archivos  ]
n = len(zoom)

px_f = []
px_t = []
Dpx_f = []
Dpx_t = []

for i in range(n):
    im_f = ndimage.rotate( of.imread( archivos[i] )[0], -0.22, reshape = False )
    perfil_f =  np.mean(  im_f[:,300:700], axis = 1  )
    corr_f = np.correlate( perfil_f - np.mean(perfil_f), perfil_f - np.mean(perfil_f), mode = 'full' )
    peaks_f = signal.find_peaks(corr_f, distance = 40*zoom[i])[0]
    px_f.append(np.mean(  np.diff(peaks_f[1:-1]) ) )
    Dpx_f.append(np.std(  np.diff(peaks_f[1:-1]) ) )

    if i == 1:
        plt.plot( corr_f ) 
        plt.plot( peaks_f, corr_f[ peaks_f.tolist() ], 'o')

    im_t = ndimage.rotate( of.imread( archivos[i] )[1], -0.22, reshape = False )
    perfil_t =  np.mean(  im_t[:,300:700], axis = 1  )
    corr_t = np.correlate( perfil_t - np.mean(perfil_t), perfil_t - np.mean(perfil_t), mode = 'full' )
    peaks_t = signal.find_peaks(corr_t, distance = 40*zoom[i])[0]
    px_t.append(np.mean( np.diff(peaks_t[1:-1]) ))
    Dpx_t.append(np.std(  np.diff(peaks_f[1:-1]) ) )


#%%  1 div = 10 um
# z2: 106.066
# z1: 212.131

plt.plot(zoom, np.array(px_f)/10, 'o')
plt.xlabel('Aumento')
plt.ylabel('pixeles por micrometro [px]')
plt.grid(True)

#%%

plt.plot(zoom, 1024/(np.array(px_f)/10), 'o')
plt.xlabel('Aumento')
plt.ylabel('Campo [um]')
plt.grid(True)
#%%

pxpum = np.array(px_f)/10
Dpxpum = np.array(Dpx_f)/10

#%%
px = np.round(1/pxpum,4)
err = np.round(0.5/pxpum**2,3)
print(px)
#%%

def recta(x, m, b):
    return m*x + b

pxpum = np.array(px_f)/10

popt, pcov = curve_fit( recta, zoom, pxpum )



plt.plot(zoom, np.array(px_f)/10, 'o')
plt.plot( zoom, recta( np.array(zoom), *popt)  )
plt.xlabel('Aumento')
plt.ylabel('pixeles por micrometro [px]')
plt.grid(True)

'''
m = (4.970+0.002)
b = (0.001+0.005)
'''


#%%

i = 1
angulos = np.arange(2, -2.7, -0.1)
px_f = []
Dpx_f = []


for ang in angulos:
    im_f = ndimage.rotate( of.imread( archivos[i] )[0], ang, reshape = False )
    perfil_f =  np.mean(  im_f[:,450:550], axis = 1  )
    corr_f = np.correlate( perfil_f - np.mean(perfil_f), perfil_f - np.mean(perfil_f), mode = 'full' )
    peaks_f = signal.find_peaks(corr_f, distance = 40*zoom[i])[0]
    px_f.append( np.mean(  np.diff(peaks_f[3:-3]) ) )
    Dpx_f.append( np.std(  np.diff(peaks_f[3:-3]) ) )

plt.plot(angulos, px_f)














