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

#%% Import files and set metadata

field = 105.6
resolution = 1600
pixel_size = field/resolution

# file_pre = "B1-R1-09-60X-pw20-k2-pre.oif"
# file_post = "B1-R1-13-60X-pw20-k2-post.oif"

file_pre = "B1-R2-11-60X-pw20-k2-pre.oif"
file_post = "B1-R2-12-60X-pw20-k2-post.oif"

# file_post = "B1-R3-14-60X-pw20-k2-post.oif"
# file_pre = "B1-R3-07-60X-pw20-k2-pre.oif"


stack_pre = of.imread( file_pre )
stack_post = of.imread( file_post )

pre1 = stack_pre[0, 2, :, :]
post0 = centrar_referencia( stack_post[0, 2, :, :] , pre1, 50)
celula = of.imread( file_pre )[1, 0, :, :]

#%% Analize correlation

im, maxi = centrar_referencia( stack_post[0, -5, :, :] , pre1, 50, maximo = True)
print(maxi)

#%% Plot 

plt.figure()
plt.imshow(post0, cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.imshow(pre1, cmap = 'gray', vmin = 80, vmax = 800)

plt.figure()
plt.imshow(np.flip( celula , 0 )  )

#%% 

plt.hist( post0.flatten(), bins = np.arange(300)*2 )


#%%

imagen = np.zeros([resolution,resolution,3])

pre_bin = np.zeros_like(pre1)
pre_bin[pre1 < 250] = 0 
pre_bin[pre1 >= 250] = 255

post_bin = np.zeros_like(post0)
post_bin[post0 < 240] = 0 
post_bin[post0 >= 240] = 255

imagen[:,:,0] = pre_bin
imagen[:,:,1] = post_bin

plt.imshow(imagen)

#%% Reconstruyo con PIV y filtro los datos con, Normalized Median Test (NMT)
vi = 128 
it = 3
suave = 1
Noise_for_NMT = 0.2
Threshold_for_NMT = 5

Y, X = n_iteraciones(suavizar(post0,suave), suavizar(pre1,suave), vi, it, bordes_extra = 6)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)

#%%
# Ploteo
l = Y_nmt.shape[0]
x,y = np.meshgrid(np.arange(l),np.arange(l))
marcas = np.arange(6)*int( np.round(field,-2)/(6-1) )
suave0 = 3
X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)

# plt.figure()
# # plt.subplot(1,3,1)
# plt.title('Resultado PIV')
# plt.yticks( marcas*pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.xticks( marcas*pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# plt.quiver(x,y,X,Y)
# #
# plt.figure()
# plt.title('Resultado NMT')
# plt.yticks( marcas*pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.xticks( marcas*pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# plt.quiver(x,y,X_nmt,Y_nmt)
# plt.subplot(1,3,2)
plt.figure()
plt.title("Suavizado")
plt.yticks( marcas/pixel_size/( vi/(2**(it-1)) ) ,marcas)
plt.xticks( marcas/pixel_size/( vi/(2**(it-1)) ) ,marcas)
plt.xlabel("Distancia [um]")
plt.ylabel("Distancia [um]")
plt.quiver(x,y,X_s,Y_s)
# plt.subplot(1,3,3)
# plt.figure()
# plt.title("Posiciones marcadas por el NMT (en blanco)")
# plt.yticks( (marcas[::-1])*pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.xticks( marcas*pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.imshow( np.fliplr(res), cmap = 'gray' )
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# # blanco son los que detecta y cambia


#%%

# r = np.sqrt(Y_nmt**2 + X_nmt**2)
# plt.figure()
# plt.title("Distribucion NMT")
# plt.xlabel('Desplazamiento [um]')
# plt.ylabel('Cuentas')
# plt.grid(True)
# plt.ylim([0,600])
# plt.hist(r.flatten()*pixel_size, bins = np.arange(-0.01,0.4, 0.02)  )
# print(np.mean( r.flatten()*pixel_size ))

X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)
r = np.sqrt(Y_s**2 + X_s**2)
r_mean = np.mean( r.flatten()*pixel_size )
plt.figure()
plt.title("Distribucion NMT suavizado, r_mean: " + str( np.round(r_mean,2)) + ' um'  )
plt.xlabel('Desplazamiento [um]')
plt.ylabel('Cuentas')
plt.grid(True)
plt.ylim([0,600])
plt.hist(r.flatten()*pixel_size, bins = np.arange(-0.01,0.4, 0.02)  )
print()


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








