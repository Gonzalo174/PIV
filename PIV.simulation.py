# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:14:37 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of

#%%
plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 15
#%%
nombre, alfa = "12_300B_60X_pw0.1_z-3_k2_rot.oif", 2048/212 # Orange 300% 60X
# nombre, alfa = "11_C4_60X_pw20_z-3_k2.oif", 2048/212 # Crimson 10% 60X

# nombre, alfa = "04_50B_40X_pw5_s-4_k2_rot.oif",  1600/318  #  Orange 50% 40X
# nombre, alfa = "08_300B_40X_pw5_z-4_k2_rot.oif",  1600/318  # Orange 300% 40X

imagen = imread(nombre)
post0 = imagen[0, 0, :, :]
post1 = imagen[0, 1, :, :]



#%%
plt.figure()
plt.imshow(post0, cmap = 'gray', vmin = 80, vmax = 800)

plt.figure()
plt.imshow(pre1, cmap = 'gray', vmin = 80, vmax = 800)

# plt.figure()
# plt.imshow(celula)

#%%
imagen = np.zeros([1600,1600,3])
imagen[:,:,0] = (post0)/np.max(post0)
imagen[:,:,1] = (pre1)/np.max(pre1)
plt.imshow(imagen)

#%%

# Simulo deformacion
tamano_de_la_deformacion = 60 #(en micrones)
pre1 = deformar( post1, 0.1, tamano_de_la_deformacion*alfa, 3 )

#%%
iio.imwrite('post0_cel1.tiff', post0)
iio.imwrite('pre1_cel1.tiff', pre1)
#%%
# Reconstruyo con PIV
vi = 128
it = 3
suave = 3

Y, X = n_iteraciones(suavizar(post0,suave), suavizar(pre1,suave), vi, it, bordes_extra = 5)

# Filtro los datos con, Normalized Median Test (NMT) con parámetros
Noise_for_NMT = 0.2
Threshold_for_NMT = 5
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)

#%%
# Ploteo
l = Y_nmt.shape[0]
x,y = np.meshgrid(np.arange(l),np.arange(l))
marcas = np.arange(5)*50
suave0 = 3
X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)
# marcas = np.array([0,50,100,150,200,250,300])
plt.figure()
# plt.subplot(1,3,1)
plt.title('Resultado PIV')
plt.yticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xlabel("Distancia [um]")
plt.ylabel("Distancia [um]")
plt.quiver(x,y,X,Y)
#
plt.figure()
plt.title('Resultado NMT')
plt.yticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xlabel("Distancia [um]")
plt.ylabel("Distancia [um]")
plt.quiver(x,y,X_nmt,Y_nmt)
# plt.subplot(1,3,2)
plt.figure()
plt.title("Suavizado")
plt.yticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xlabel("Distancia [um]")
plt.ylabel("Distancia [um]")
plt.quiver(x,y,X_s,Y_s)
# plt.subplot(1,3,3)
plt.figure()
plt.title("Posiciones marcadas por el NMT (en blanco)")
plt.yticks( (marcas[::-1])*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.xticks( marcas*alfa/( vi/(2**(it-1)) ) ,marcas)
plt.imshow( np.fliplr(res), cmap = 'gray' )
plt.xlabel("Distancia [um]")
plt.ylabel("Distancia [um]")
# blanco son los que detecta y cambia


#%%

r = np.sqrt(Y_nmt**2 + X_nmt**2)
plt.figure()
plt.title("Distribucion NMT")
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
plt.ylim([0,460])
plt.hist(r.flatten()/alfa, bins = np.arange(-0.01,0.7, 0.02)  )
print(np.mean( r.flatten()/alfa ))

X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)
r = np.sqrt(Y_s**2 + X_s**2)
plt.figure()
plt.title("Distribucion NMT suavizado")
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
plt.ylim([0,460])
plt.hist(r.flatten()/alfa, bins = np.arange(-0.01,0.7, 0.02)  )
print(np.mean( r.flatten()/alfa ))

'''

En un futuro estaria bueno promediar solo en las areas de las adhesiones
'''
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








