# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:45:37 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = "Times New Roman"

#%%

cm_crimson = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255)) for i in range(1000)] )
cm_orange = ListedColormap( [(i/(999),165*i/(999*255),0) for i in range(1000)] )
cm_green = ListedColormap( [(0,128*i/(999*255),0) for i in range(1000)] )


#%% Enfoque con la altura 

C22 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Primer plano\cel22_3X.tif" )[:,0]
# C004 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\C_0.004.tif" )[-3,0]
#%%
plt.imshow( C22[6], cmap = cm_crimson, vmin = 150, vmax = 700 )


#%%
lC = 16 # um
lC0 = 11
ps = 0.067
dC0 = int(lC0/ps)
dC = int(lC/ps)
Vm0 = 600
z = 8

scale_length = 1  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar
# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = int(3.7/ps)  # Starting x-coordinate of the scale bar
start_y = int(6/ps)  # Starting y-coordinate of the scale bar

plt.figure(figsize=[5,5], tight_layout=True)
# plt.subplot(1,2,1)
# plt.title('0.0002%')
plt.imshow( C22[z,int(dC0 + 4/ps):int(dC + 4//ps),dC0:dC], cmap = cm_crimson, vmin = 150, vmax = Vm0 )

for i in np.arange(-13,-10, 0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 25, f'{scale_length} {scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.text(start_x + scale_pixels/2 + 5, 10, str(z), color='white', weight='bold', ha='center', fontsize = "xx-large")
plt.xticks([])
plt.yticks([])

plt.show()

#%%
plt.rcParams['font.size'] = 16

lC = 16 # um
lC0 = 11
ps = 0.067
dC0 = int(lC0/ps)
dC = int(lC/ps)
Vm0 = 600
z = 8

scale_length = 1  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar
# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = int(3.5/ps)  # Starting x-coordinate of the scale bar
start_y = int(5.9/ps)  # Starting y-coordinate of the scale bar

plt.figure(figsize=[27,4], tight_layout=True)
for z in range(1,8,1):
    plt.subplot( 1, 7, z  )

    plt.imshow( C22[z,int(dC0 + 4/ps):int(dC + 4//ps),dC0:dC], cmap = cm_crimson, vmin = 150, vmax = Vm0 )
    plt.imshow( C22[z-1,int(dC0 + 4/ps) + 10:int(dC + 4//ps)+10,dC0+5:dC+5], cmap = cm_green, vmin = 150, vmax = Vm0, alpha = 0.5 )

    
    for i in np.arange(-12,-9, 0.1):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
    if z == 1:
        plt.text(start_x + scale_pixels/2, start_y - 25, f'{scale_length} {scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
    # plt.text(start_x + scale_pixels/2 - 10, 8, 'Z = ' + str((int(-(z/2-2)*10)/10)) + ' ' + f'{scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
    plt.xticks([])
    plt.yticks([])

#%%

lC = 16*2 # um
lC0 = 11
ps = 0.067
dC0 = int(lC0/ps)
dC = int(lC/ps)
Vm0 = 600
z = 8

scale_length = 5  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar
# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = int(14/ps)  # Starting x-coordinate of the scale bar
start_y = int(20/ps)  # Starting y-coordinate of the scale bar

plt.figure(figsize=[27,4], tight_layout=True)
for z in range(1,8,1):
    plt.subplot( 1, 8, z )

    plt.imshow( C22[z,int(dC0 + 4/ps):int(dC + 4//ps),dC0:dC], cmap = cm_crimson, vmin = 150, vmax = Vm0 )
    
    for i in np.arange(-5,5, 0.1):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
    if z == 1:
        plt.text(start_x + scale_pixels/2, start_y - 25, f'{scale_length} {scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
    plt.text(start_x + scale_pixels/2 - 30, 40, 'Z = ' + str((int(-(z/2-1)*10)/10)) + ' ' + f'{scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
    plt.xticks([])
    plt.yticks([])    
 
    
#%%
plt.rcParams['font.size'] = 30
intens = [np.std( C22[z] ) for z in range(1,8,1) ]
intens2 = normalizar( intens )

intens_post = [np.std( C22[z] ) for z in range(0,7,1) ]
intens2_post = normalizar( intens_post )


plt.figure( figsize = [27,6] )
plt.plot([(int(-(z/2-1)*10)/10) for z in range(1,8,1)], intens2, 'o', ms = 10, c = (220/255,20/255,60/255), label = 'PRE'  )
plt.plot([(int(-(z/2-1)*10)/10) for z in range(1,8,1)], intens2_post, 'o', ms = 10, c = (0.08,0.7,0.01), label = 'POST'  )
plt.legend()
plt.grid()

plt.xlim([0.7, -2.7])
plt.xlabel( 'Profundidad [µm]' )
plt.ylabel( 'Desvío estandar [U.A.]' )
plt.yticks([0,0.25,0.5,0.75,1])
plt.show()
#%%






















os.listdir(r'C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas')

#%%

C0002 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\C_0.0002.tif" )[4,0]
C004 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\C_0.004.tif" )[-3,0]

O04 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\O_0.04.tif" )[0,0]
O04_60X = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\O_0.04_60X.tif" )[-1,0]

O6 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\O_0.6.tif" )[-1,0]
O6_60X = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\O_0.6_60X.tif" )[-1,0]

O2 = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\O_0.2.tif" )[-3,0]

#%% Crimson 60X
lC = 30 # um
ps = 105.47/1024
dC = int(lC/ps)
Vm0 = 600

scale_length = 3  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar
# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = int(25/ps)  # Starting x-coordinate of the scale bar
start_y = int(29.5/ps)  # Starting y-coordinate of the scale bar

plt.figure(tight_layout=True)
plt.subplot(1,2,1)
# plt.title('0.0002%')
plt.imshow( C0002[300:300+dC,300:300+dC], cmap = cm_crimson, vmin = 150, vmax = Vm0 )
for i in np.arange(-5,5,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 25, f'{scale_length} {scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.text(start_x + scale_pixels/2 - 10, 25, '0.0002%', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
# plt.title('0.004%')
plt.imshow( C004[:dC,:dC], cmap = cm_crimson, vmin = 150, vmax = Vm0 )
for i in np.arange(-5,5,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2 - 5, 25, '0.004%', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.xticks([])
plt.yticks([])

plt.show()


#%% Orange 40X
l40X = 30 # um
ps = 316.8/1600
d40X = l40X/ps
Vm = 3000

scale_length = 3  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar
# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = int(24.7/ps)  # Starting x-coordinate of the scale bar
start_y = int(31.2/ps)  # Starting y-coordinate of the scale bar

plt.figure(figsize = [14, 12], tight_layout=True)

plt.subplot(1,3,1)
# plt.title('0.04%')
plt.imshow( O04[:int(d40X),:int(d40X)], cmap = cm_orange, vmin = 70, vmax = Vm )
for i in np.arange(-8,-2,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 25, f'{scale_length} {scale_unit}', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.text(start_x + scale_pixels/2 - 2, 14, '0.04%', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
# plt.title('0.2%')
plt.imshow( O2[:int(d40X),:int(d40X)], cmap = cm_orange, vmin = 70, vmax = Vm )
for i in np.arange(-8,-2,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2, 14, '0.2%', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
# plt.title('0.6%')
plt.imshow( O6[:int(d40X),:int(d40X)], cmap = cm_orange, vmin = 70, vmax = Vm )
for i in np.arange(-8,-2,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2, 14, '0.6%', color='white', weight='bold', ha='center', fontsize = "x-large")
plt.xticks([])
plt.yticks([])

plt.show()

#%% Orange 60X
l60X = 30 # um
ps = 316.8/1600
d60X = l60X/ps
Vm2 = 2000

scale_length = 3  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar
# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = int(25/ps)  # Starting x-coordinate of the scale bar
start_y = int(31.8/ps)  # Starting y-coordinate of the scale bar

plt.figure(figsize = [12, 24], tight_layout=True)

plt.subplot(1,2,1)
# plt.title('0.04%')
plt.imshow( O04_60X[200:200+int(d60X),50:50+int(d60X)], cmap = cm_orange, vmin = 70, vmax = Vm2 )
for i in np.arange(-9,-4,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 25, f'{scale_length} {scale_unit}', color='white', weight='bold', ha='center', fontsize = "xx-large")
plt.text(start_x + scale_pixels/2, 12, '0.04%', color='white', weight='bold', ha='center', fontsize = "xx-large")
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
# plt.title('0.6%')
plt.imshow(  O6_60X[:int(d60X),:int(d60X)], cmap = cm_orange, vmin = 70, vmax = Vm2 )
for i in np.arange(-9,-4,0.1):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)
plt.text(start_x + scale_pixels/2 + 3, 12, '0.6%', color='white', weight='bold', ha='center', fontsize = "xx-large")
plt.xticks([])
plt.yticks([])

plt.show()


#%%
l60X = 30 # um
d60X = l60X*1600/316.80
Vm2 = 2000

plt.figure()
plt.title('Orange 0.04% 60X')
plt.imshow( O04_60X[200:200+int(d60X),50:50+int(d60X)], cmap = 'gray', vmin = 70, vmax = Vm2 )

plt.figure()
plt.title('Orange 0.6% 60X')
plt.imshow( O6_60X[:int(d60X),:int(d60X)], cmap = 'gray', vmin = 70, vmax = Vm2 )












#%%

stack = iio.imread( r"C:\Users\gonza\1\Tesis\Tesis\Mediciones aisladas\Tipo y concentración de nanoesferas\C_0.004.tif" )[:,0]
pre = stack[-4]
ps = 0.1007

plt.figure()
plt.imshow( pre, cmap = 'gray', vmin = 70, vmax = 700 )

#%%
a, b = 4, 9
w = 60
pre_chico = pre[ int(w*a+2) : int(w*(a+1)+2), int(w*b+2) : int(w*(b+1)+2) ]

plt.figure()
plt.imshow( pre_chico, cmap = 'gray', vmin = 100, vmax = 700 )

#%%
a, b = 4, 8
w = 15
l = int(1024/w)
desvios = np.zeros( [l]*2 )

for j in range(l):
    for i in range(l):
        pre_chico = pre[ int(w*j+2) : int(w*(j+1)+2), int(w*i+2) : int(w*(i+1)+2) ]
        desvios[j,i] = np.std( pre_chico )

plt.imshow(desvios)
plt.colorbar()





#%% 20/8

# a, b = 1.5,4.5
a, b = 4, 8
w = 64
a2 = np.mean(pre)/np.mean(post)

pre1_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))]*a2 


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



#%% Nanospheres intensity histogram
val_pre = pre.flatten()
val_post = post.flatten()

plt.figure()
plt.hist( (val_pre), bins = np.arange(80,700,2), density = True)
plt.title('pre')
plt.ylim([0,0.02])

plt.figure()
plt.hist(val_post, bins = np.arange(80, 700, 2), density = True)
plt.title('post')
plt.ylim([0,0.02])

#%%

a, b = 12, 5
w = 16
a2 = np.mean(pre)/np.mean(post)

pre1_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))]*a2 


plt.figure()
plt.subplot(1,2,1)
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.subplot(1,2,2)
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)


val_pre = pre.flatten()
val_pre_chico = pre1_chico.flatten()

plt.figure()
plt.hist( (val_pre), bins = np.arange(80,700,2), density = True)
plt.hist( (val_pre_chico), bins = np.arange(80,700,20), density = True)

plt.title('pre')
plt.ylim([0,0.02])



#%%

inf = 120
a = np.mean(post)/np.mean(pre)

pre_plot = np.copy( (pre+5)*a - inf )
post_plot = np.copy(post - inf )

pre_plot[ pre < 0 ] = 0
pre_plot[ post < 0 ] = 0

c0 = [(0, 0, 0), (0, 0, 0)]
cm0 = ListedColormap(c0)

c1 = []
c2 = []
for i in range(1000):
    c1.append((i/999,0,0))
    c2.append((0,i/999,0))

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)


plt.figure()
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)


#%%

from matplotlib.colors import ListedColormap
colors = [(0, 0, 0),  # Black
          (1, 0, 0),  # Red
          (1, 1, 0),  # Yellow
          (0, 1, 0),  # Green
          (0, 0, 1)]  # Blue


#%%
prueba = np.ones([2,2])

prueba[:,0] = [0,0]

plt.imshow(prueba, cmap = cm1, alpha = 1)
plt.imshow(prueba.T, cmap = cm2, alpha = 0.5)


#%% 12/10

val_1 = np.copy(pre1[512:].flatten())
val_2 = np.copy(pre1[:512].flatten())

plt.figure()
plt.hist( val_2, bins = np.arange(2050)*2, density = False, label = "2")
plt.hist( val_1, bins = np.arange(2050)*2, density = False, label = "1")
# plt.xlim([0,2000])
plt.yscale("log")
plt.legend()
plt.grid(True)


#%%

um = 800

res = np.zeros( pre.shape )
res[ pre > um ] = 1
pre1 = np.copy( pre )
for i in range(3):
    res = area_upper(res, 10)
pre1[ res == 1 ] = np.mean( pre*(1-res)*(1024**2)/(1024**2 - np.sum(res)) )

plt.imshow(pre1)

#%%

um = 800

res = np.zeros( post.shape )
res[ post > um ] = 1
post1 = np.copy( post )
post1[ post > um ] = np.mean( post*(1-res)*(1024**2)/(1024**2 - np.sum(res)) )

plt.imshow(post1)



#%% 13/10

plt.figure()
plt.title('Pre')
plt.imshow( pre , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post, cmap = 'gray', vmin = 80, vmax = 700)

#%%
a, b = 16, 23
w = 32
a2 = np.mean(pre)/np.mean(post)

pre1_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))]*a2 

plt.figure()
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)


#%%

w = 32
l = 1024//w
std = np.zeros([ l ]*2)

for a in range(l):
    for b in range(l):
        std[a,b] = np.std( pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ] )


plt.figure()
plt.title('Pre')
plt.imshow( pre , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('std')
plt.imshow(std, vmax = 150)
plt.colorbar()


#%%

plt.hist( std.flatten(), bins = np.arange(40, 400, 10) )



#%%

plt.hist( stack_pre[-1].flatten(), bins = np.arange(0, 400, 2) )







