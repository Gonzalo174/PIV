# -*- coding: utf-8 -*-
"""
Created on Thu May  4 23:14:37 2023

@author: gonza
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy import ndimage
from scipy import signal
from scipy import optimize
from scipy.optimize import curve_fit
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [9,9]
plt.rcParams['font.size'] = 15

#%%

# Specify the folder path
folder_path = r'C:\Users\gonza\1\Tesis\2023\23.05.18 - Celulas en geles OA Orange'  # Replace with the path to your folder
# folder_path = r'C:\Users\gonza\1\Tesis\2022\practica-PIV\gel8'

# Get the list of file names in the folder
file_names = os.listdir(folder_path)

# Print the file names
for file_name in file_names:
    print(file_name)


#%% Import files and set metadata
# field = 75.2
field = 105.6
resolution = 1600
pixel_size = field/resolution

# Viejos

# file_cell = "gel1_cell1_TFM_47nm_PRE_K.oif"
# file_pre = "gel1_cell1_TFM_47nm_PRE_K.oif"
# file_post = "gel1_cell1_TFM_47nm_Post_K.oif"

# Orange 18/5

# file_cell = "09-BOB-R2-60X-pw0.2-k0-zoomX2-tra.oif"
# file_pre = "10-BOB-R2-60X-pw0.2-k2-zoomX2-pre.oif"
# file_post = "13-BOB-R2-60X-pw0.2-k2-zoomX2-post.oif"

# file_cell = "01-BOA-R1-60X-pw0.1-k0-tra.oif"
# file_pre = "02-BOA-R1-60X-pw0.2-k0-pre.oif"
# file_post = "06-BOA-R1-60X-pw0.2-k2-post.oif"

# file_cell = "03-BOA-R2-60X-pw0.2-k0-tra.oif"
# file_pre = "04-BOA-R2-60X-pw0.2-k2-pre.oif"
# file_post = "05-BOA-R2-60X-pw0.2-k2-post.oif"

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

#%% Analize correlation

n = 0
pre1 = stack_pre[ n ]
post0 = centrar_referencia( stack_post[ n ] , pre1, 250)
# post0 = centrar_referencia_3D( stack_post, pre1, 250)


#%%
mascara1 = iio.imread( "celula_Crimson_R3_cell1.png" )
mascara2 = iio.imread( "celula_Crimson_R3_cell2.png" )
mascara3 = iio.imread( "celula_Crimson_R3_cell3.png" )
mascara4 = iio.imread( "celula_Crimson_R3_cell4.png" )
mascara5 = iio.imread( "celula_Crimson_R3_cell5.png" )


#%%
# pre1 = pre1[750:1450, 800:1500]
# post0 = post0[750:1450, 800:1500]
# celula = celula[750:1450, 800:1500]

#%% Varias alturas

#pre1 = stack_pre[0, 0] + stack_pre[0, 1] + stack_pre[0, 2]
#post0 = centrar_referencia( stack_post[0, 0] + stack_post[0, 1] + stack_post[0, 2], pre1, 50)

#%% Plot 

plt.figure()
plt.title('Pre')
plt.imshow( np.flip( pre1, 0 ) , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow(  np.flip( post0, 0 ) , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
# plt.title('Trans')
plt.imshow(np.flip( celula , 0 )  )
plt.axis('off')

#%% Ventanas de exploracion

# a, b = 1.5,4.5
a, b = 10, 5
w = 32

pre1_chico = pre1[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post0[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 


plt.figure()
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)


plt.figure()
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)

# plt.figure()
# plt.title('Post')
# plt.imshow(post0, cmap = 'gray', vmax = 400)
# plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )

# plt.figure()
# plt.title('Trans')
# plt.imshow(celula, cmap = 'gray')
# plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )


#%%
plt.rcParams['font.size'] = 21

borde = 5
pre_win = pre1[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post_bigwin = post0[int(w*a)-borde : int(w*(a+1))+borde, int(w*b)-borde : int(w*(b+1))+borde] 

cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method="fft")
# cross_corr = suavizar( cross_corr, 3 )

y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
y, x = -(y0 - borde), -(x0 - borde)

u, v = np.meshgrid(np.linspace(-borde, borde, 2*borde+1), np.linspace(-borde, borde, 2*borde+1))
data = cross_corr.ravel()
initial_guess = [np.max(data)-np.min(data), -x, -y, 3, 3, 0, np.min(data)]
popt, pcov = curve_fit(gaussian_2d, (u, v), data, p0=initial_guess)
amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
gauss_fit = gaussian_2d_plot( (u,v), *popt  )

print(x,y)
print(-xo,-yo)
print( np.mean( cross_corr**2 - gauss_fit**2 ) )

#%%
plt.figure()
plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 + 1 - borde) )
plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 + 1 - borde )
plt.xlabel("Distancia [px]")
plt.ylabel("Distancia [px]")
plt.title("Correlación cruzada")
# plt.text( borde+x, y0+1.2, f'({x},{y})', color='r', weight='bold', ha='center')
plt.imshow( np.flip(cross_corr,1) )# , vmin = 0, vmax = 500000 )
plt.plot( [borde+x], [y0], 'o',c = 'red', markersize = 10 )
plt.plot( [borde-xo], [borde+yo], 'o',c = 'green', markersize = 10 )
plt.colorbar()

# print( signal.correlate(post0 - post0.mean(), pre1 - pre1.mean(), mode = 'valid', method="fft")[0][0]/(1600**2)  )
# print(np.mean(cross_corr[y0-1:y0+2,x0-1:x0+2])/(w**2))

#%%

plt.figure()
plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 +1- borde) )
plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 +1- borde )
plt.xlabel("Distancia [px]")
plt.ylabel("Distancia [px]")
plt.title("Correlación cruzada - Ajuste")

# gauss_fit = gaussian_2d_plot( (u,v), *initial_guess  )

plt.imshow(  np.flip(gauss_fit,1) )
plt.plot( [borde+x], [y0], 'o',c = 'red', markersize = 10 )
plt.plot( [borde-xo], [borde+yo], 'o',c = 'green', markersize = 10 )
plt.colorbar()


#%%

plt.plot( cross_corr[:,-7] )
plt.plot( gauss_fit[:,-7] )












#%%
plt.rcParams['font.size'] = 21
borde = 10
 
pre_win =      pre1[ int(w*a)        : int(w*(2*a+1)/2)         , int(w*b)                  : int(w*(2*b+1)/2)        ]
post_bigwin = post0[ int(w*a)-borde  : int(w*(2*a+1)/2) + borde , int(w*b) - borde          : int(w*(2*b+1)/2) + borde] 

# pre_win =      pre1[ int(w*a)         : int(w*(2*a+1)/2)          , int(w*(2*b+1)/2)         : int(w*(b+1))        ]
# post_bigwin = post0[ int(w*a) - borde : int(w*(2*a+1)/2) + borde  , int(w*(2*b+1)/2) - borde : int(w*(b+1)) + borde] 

# pre_win =      pre1[ int(w*(2*a+1)/2)        : int(w*(a+1))       , int(w*b)                  : int(w*(2*b+1)/2)        ]
# post_bigwin = post0[int(w*(2*a+1)/2) - borde : int(w*(a+1)) + borde , int(w*b) - borde          : int(w*(2*b+1)/2) + borde] 

# pre_win =      pre1[ int(w*(2*a+1)/2)        : int(w*(a+1))    ,int(w*(2*b+1)/2)         : int(w*(b+1))       ]
# post_bigwin = post0[int(w*(2*a+1)/2) - borde : int(w*(a+1))+borde, int(w*(2*b+1)/2) - borde : int(w*(b+1)) + borde] 

cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method="fft")
cross_corr = suavizar( cross_corr, 3 )
y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
y, x = -(y0 - borde), -(x0 - borde)

print(x,y)

plt.figure()
plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 +1- borde) )
plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 +1- borde )
plt.xlabel("Distancia [px]")
plt.ylabel("Distancia [px]")
plt.title("Correlación cruzada")
plt.text( borde+x, y0+1.2, f'({x},{y})', color='r', weight='bold', ha='center')
plt.imshow( np.flip(cross_corr,1) )
plt.plot( [borde+x], [y0], 'o',c = 'red', markersize = 10 )







#%% Reconstruyo con PIV y filtro los datos con, Normalized Median Test (NMT)
vi = 128
it = 3
exploration = 8 # px
suave = 1
Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Fit"
mapas = True

Y, X = n_iteraciones(suavizar(post0,suave), suavizar(pre1,suave), vi, it, bordes_extra = exploration, mode = modo)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)

#%% Ploteo
l = Y_nmt.shape[0]
x,y = np.meshgrid(np.arange(l),np.arange(l))
marcas = np.arange(6)*int( np.round(field,-2)/(6-1) )
suave0 = 3
scale0 = 250
X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)

# plt.figure()
# # plt.subplot(1,3,1)
# plt.title('Resultado PIV - ' + modo)
# plt.yticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
# plt.xticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# plt.quiver(x,y,X,Y, scale = scale0)

plt.figure()
plt.title('Resultado NMT - ' + modo)
plt.yticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
plt.xticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
plt.xlabel("Distancia [um]")
plt.ylabel("Distancia [um]")
plt.quiver(x,y,X_nmt,Y_nmt, scale = scale0)

# plt.subplot(1,3,2)
# plt.figure()
# plt.title("Suavizado - " + modo)
# plt.yticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
# plt.xticks( marcas/pixel_size/( vi/(2**(it-1)) ) , marcas)
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# plt.quiver(x,y,X_s,Y_s, scale = scale0)
# plt.subplot(1,3,3)
# plt.figure()
# plt.title("Posiciones marcadas por el NMT (en blanco)")
# plt.yticks( (marcas[::-1])/pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.xticks( marcas/pixel_size/( vi/(2**(it-1)) ) ,marcas)
# plt.imshow( np.fliplr(res), cmap = 'gray' )
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# blanco son los que detecta y cambia
# plt.figure()
# plt.title("Célula")
# plt.yticks( marcas/pixel_size  ,marcas)
# plt.xticks( marcas/pixel_size  ,marcas)
# plt.xlabel("Distancia [um]")
# plt.ylabel("Distancia [um]")
# plt.imshow( np.flip( celula , 0 ) )


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

X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)
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




#%% 

plt.hist( post0.flatten(), bins = np.arange(300)*2 )


#%%

imagen = np.zeros([resolution,resolution,3])

th0, th = 100, 400
# pre_bin = np.zeros_like(pre1)
pre_bin = np.copy( (pre1-th0)/th )
pre_bin[suavizar(pre1,3) < th] = 0 
pre_bin[suavizar(pre1,3) >= th] = 1

# post_bin = np.zeros_like(post0)
post_bin = np.copy( (post0-th0)/th )
post_bin[suavizar(post0,3) < th0] = 0 
post_bin[suavizar(post0,3) >= th] = 1

imagen[:,:,1] = pre_bin
imagen[:,:,0] = post_bin

plt.figure()
plt.imshow( np.flip( celula , 0 )  )
# plt.xlim([700,1200])
# plt.ylim([1000,1400])
plt.figure()
plt.imshow( np.flip(imagen,0) )
# plt.xlim([700,1200])
# plt.ylim([1000,1400])






