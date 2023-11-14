# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:48:00 2023

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

cm0 = ListedColormap( [(0, 0, 0), (0, 0, 0)] )               # Negro
cm1 = ListedColormap( [(i/999,0,0) for i in range(1000)] )   # Negro - Rojo
cm2 = ListedColormap( [(0, i/999, 0) for i in range(1000)] )   # Negro - Verde
cm3 = ListedColormap( [(1, 1, 1), (1, 1, 0)] )               # Blanco - Amarillo
cm_orange = ListedColormap( [(i/(999),165*i/(999*255),0) for i in range(1000)] )   # Negro - Naranja
cm_orange2 = ListedColormap( [(1,1,1), (1,165/255,0)] )   # Blanco - Naranja


#%% Import
path = r"C:\Users\gonza\1\Tesis\2023\\" + "23.10.05 - gon MCF10 2 - D04"
# path = r"D:\Gonzalo\\" + "23.10.20 - gon MCF10 7 - I18"

#%%
mediciones = os.listdir(path)
mediciones8 = [ medicion for medicion in mediciones if '8' in medicion and 'files' not in medicion]
print( mediciones8 )
#%%

stack_pre1 = of.imread( path + r"\\" + mediciones8[1] )[0]
stack_pre2 = of.imread( path + r"\\" + mediciones8[3] )[0]
stack_pre3 = of.imread( path + r"\\" + mediciones8[4] )[0]

stack_post1 = of.imread( path + r"\\" + mediciones8[6] )[0]

cel_pre1 = of.imread( path + r"\\" + mediciones8[1] )[1,2]
cel_pre2 = of.imread( path + r"\\" + mediciones8[3] )[1,2]
cel_pre3 = of.imread( path + r"\\" + mediciones8[4] )[1,2]
m1 = iio.imread( path + r"\\" + 'cel11_m1.png' )
m2 = iio.imread( path + r"\\" + 'cel11_m2.png' )
m3 = iio.imread( path + r"\\" + 'cel11_m3.png' )


cel_post1 = of.imread( path + r"\\" + mediciones8[6] )[1,1]

#%%

plt.imsave( path + r"\\" +'cel11_1.png', cel_pre1, cmap = 'gray')
plt.imsave( path + r"\\" +'cel11_2.png', cel_pre2, cmap = 'gray')
plt.imsave( path + r"\\" +'cel11_3.png', cel_pre3, cmap = 'gray')
# plt.imsave('cel10_4.png', cel_pre4, cmap = 'gray')
ps = 0.1007
resolution = 1024
# pre_list = [3, 5, 5]

#%%
pre = stack_pre1[4]
post = stack_post1[3]

a = np.mean(post)/np.mean(pre)

plt.figure()
plt.title('pre')
plt.imshow( pre*a , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('post')
plt.imshow( post , cmap = 'gray', vmin = 80, vmax = 700)

#%%
post = stack_post1[3]
pre, ZYX = correct_driff_3D( stack_pre3, post, 50, info = True )
c = cel_pre3
m = m3
N = 3

# ZYX = (0, 50, 50)
celula_pre = np.ones( [1024]*2 )*np.mean(c)
celula_pre[ max( 0, ZYX[1] ) : min( 1024, 1024 + ZYX[1]) , max( 0, ZYX[2]) : min( 1024, 1024 + ZYX[2])  ] = c[ max( 0, -ZYX[1] ) : min( 1024, 1024 + -ZYX[1])  ,  max( 0, -ZYX[2]) : min( 1024, 1024 + -ZYX[2])  ]

mascara = np.zeros( [1024]*2 )
mascara[ max( 0, ZYX[1] ) : min( 1024, 1024 + ZYX[1]) , max( 0, ZYX[2]) : min( 1024, 1024 + ZYX[2])  ] = m[ max( 0, -ZYX[1] ) : min( 1024, 1024 + -ZYX[1])  ,  max( 0, -ZYX[2]) : min( 1024, 1024 + -ZYX[2])  ]

# m3_pro = np.copy(mascara)

# PIV + NMT + Suavizado
vi = int( int( 3/ps )*4 )
it = 3
# bordes_extra = 10 # px
bordes_extra = int(np.round(vi/12))

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth5"
# modo = "No control"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( pre, post, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
# mascara = 1 - iio.imread( "mascara_3.png" )

x, y = dominio

# inf = 120
# a = np.mean(post)/np.mean(pre)
# pre_plot = np.copy( (pre+5)*a - inf )
# post_plot = np.copy(post - inf )
# pre_plot[ pre < 0 ] = 0
# post_plot[ post < 0 ] = 0

scale0 = 100
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar

wind = vi/( 2**(it-1) )
d = int( ( resolution - len(Y_nmt)*wind )/2   )

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + 100  # Starting x-coordinate of the scale bar
start_y = resolution -( 2*wind ) + 10# Starting y-coordinate of the scale bar


plt.figure(figsize=(20,20), tight_layout=True)

plt.subplot(1,3,1)

plt.imshow( celula_pre , cmap = 'gray' )
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 35, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")
plt.text(start_x, 100, str(N), color='black', weight='bold', ha='center', fontsize = "xx-large")


plt.subplot(1,3,2)

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = cm_orange2, alpha = 0.4 )
plt.quiver(x,y,-X_s,Y_s, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.subplot(1,3,3)
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
# plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
# plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)

plt.imshow( pre, cmap = cm1, vmin = 120, vmax = 400, alpha = 1)
plt.imshow( post, cmap = cm2, vmin = 120, vmax = 400, alpha = 0.5)

plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)


# plt.savefig(name + '_figura.png')
plt.show()


#%%


plt.figure()
plt.imshow(m4_grande)


#%%
y0 = 600
vecinos = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
# mascara = m1_pro
mascara = np.zeros([1024]*2)
mascara[ smooth(m1_pro, 3) > 0.5 ] = 1

y_borde = [y0]
x_borde = []
j = 0
while len(x_borde) == 0:
    if mascara[y0,j] == 1:
        x_borde.append(j-1)
    j += 1    


# while (y_borde[-1] - y_borde[0])**2 + (x_borde[-1] - x_borde[0])**2 <= 1 and len( x_borde ) > 1000:
seguir = True
while seguir:
    ultim0_vecino = [0,0]
    x0 = x_borde[-1] 
    y0 = y_borde[-1]
    for j in range(8):
        v0 = mascara[ y0 + vecinos[j-1][0], x0 + vecinos[j-1][1] ]
        v1 = mascara[   y0 + vecinos[j][0],   x0 + vecinos[j][1] ]
        if v0 == 0 and v1 == 1:
            x_borde.append( x0 + vecinos[j-1][1] )
            y_borde.append( y0 + vecinos[j-1][0] )
            ultimo_vecino = vecinos[j-1]
            
    if len(x_borde) > 4:
        if ( x_borde[-1] == x_borde[-2] and y_borde[-1] == y_borde[-2] ) or ( x_borde[-1] == x_borde[-3] and y_borde[-1] == y_borde[-3] ) or ( x_borde[-1] == x_borde[-4] and y_borde[-1] == y_borde[-4] ):
            x_borde.append( x0 + ultimo_vecino[1] )
            y_borde.append( y0 + ultimo_vecino[0] )
        
    if ( x_borde[-1] == x_borde[0] and y_borde[-1] == y_borde[0] and len(x_borde) > 1 ) or len(x_borde) > 10000:
        seguir = False


borde1 = np.concatenate( (  np.reshape( np.array( y_borde ), [1, len(y_borde)]) , np.reshape( np.array( x_borde ) ,  [1, len(y_borde)] ) ) , axis = 0 )


#%%

plt.imshow( 1-m1_pro, cmap='gray' , alpha = 0.1)
plt.plot(borde1[1],borde1[0],c = 'b', linewidth = 2 )

#%%
plt.figure()
plt.imshow( m1_pro, cmap='gray' , alpha = 0.01)
plt.plot(borde1[1],borde1[0],c = (0.298, 0.447, 0.690), linewidth = 2, label = "T1" )
plt.plot(borde2[1],borde2[0],c = (0.866, 0.517, 0.321), linewidth = 2, label = "T2" )
plt.plot(borde3[1],borde3[0],c = (0.333, 0.658, 0.407), linewidth = 2, label = "T3" )
# plt.plot(borde4[1],borde4[0],c = (0.768, 0.305, 0.321), linewidth = 2, label = "T4" )
plt.legend()


plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 35, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")



