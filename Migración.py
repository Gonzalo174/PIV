# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:39:09 2023

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
path = r"C:\Users\gonza\1\Tesis\2023\\" + "23.10.20 - gon MCF10 7 - I18"
# path = r"D:\Gonzalo\\" + "23.10.20 - gon MCF10 7 - I18"

#%%
mediciones = os.listdir(path)
mediciones26 = [ medicion for medicion in mediciones if '26' in medicion and 'files' not in medicion]
print( mediciones26 )
#%%

stack_pre1 = of.imread( path + r"\\" + mediciones26[1] )[0]
stack_pre2 = of.imread( path + r"\\" + mediciones26[3] )[0]
stack_pre3 = of.imread( path + r"\\" + mediciones26[4] )[0]
stack_pre4 = of.imread( path + r"\\" + mediciones26[5] )[0]

stack_post1 = of.imread( path + r"\\" + mediciones26[7] )[0]
stack_post2 = of.imread( path + r"\\" + mediciones26[6] )[0]

cel_pre1 = of.imread( path + r"\\" + mediciones26[1] )[1,1]
cel_pre2 = of.imread( path + r"\\" + mediciones26[3] )[1,1]
cel_pre3 = of.imread( path + r"\\" + mediciones26[4] )[1,1]
cel_pre4 = of.imread( path + r"\\" + mediciones26[5] )[1,1]
m1 = iio.imread( path + r"\\" + 'cel10_m1.png' )
m2 = iio.imread( path + r"\\" + 'cel10_m2.png' )
m3 = iio.imread( path + r"\\" + 'cel10_m3.png' )
m4 = iio.imread( path + r"\\" + 'cel10_m4.png' )


cel_post1 = of.imread( path + r"\\" + mediciones26[7] )[1,1]
cel_post2 = of.imread( path + r"\\" + mediciones26[6] )[1,1]

#%%

# plt.imsave('cel10_1.png', cel_pre1, cmap = 'gray')
# plt.imsave('cel10_2.png', cel_pre2, cmap = 'gray')
# plt.imsave('cel10_3.png', cel_pre3, cmap = 'gray')
# plt.imsave('cel10_4.png', cel_pre4, cmap = 'gray')

#%%
ps = 0.1007

p1 = stack_post1[5]
p2 = stack_post2[5]

p2_c, M, YX = correct_driff( p2, p1, 300, info = True )

#%%
plt.figure()
plt.title('P2')
plt.imshow( p2_c , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('P1')
plt.imshow( p1 , cmap = 'gray', vmin = 80, vmax = 700)


#%% (-97, 195) p2 está desplazado hacia los positivos en X y hacia los negativos en Y con respectos a p1
lz = len(stack_post1)
post_grande = np.ones(  [ lz, 1219, 1219 ]  )*np.mean(stack_post1)

post_grande[ :, :1024, -1024: ] = stack_post1
post_grande[ :, 97:1024+97, -1024-195:-195 ] = stack_post2

plt.figure()
plt.title('P1')
plt.imshow( post_grande[5] , cmap = 'gray', vmin = 80, vmax = 700)

resolution = 1219
#%%

pre_grande1 = np.ones(  [ len(stack_pre1), 1219, 1219 ]  )*np.mean(stack_pre1)
pre_grande1[ :, 100:1124, -1124:-100 ] = stack_pre1
pre_grande2 = np.ones(  [ len(stack_pre2), 1219, 1219 ]  )*np.mean(stack_pre1)
pre_grande2[ :, 100:1124, -1124:-100 ] = stack_pre2
pre_grande3 = np.ones(  [ len(stack_pre3), 1219, 1219 ]  )*np.mean(stack_pre1)
pre_grande3[ :, 100:1124, -1124:-100 ] = stack_pre3
pre_grande4 = np.ones(  [ len(stack_pre4), 1219, 1219 ]  )*np.mean(stack_pre1)
pre_grande4[ :, 100:1124, -1124:-100 ] = stack_pre4

#%%
plt.figure()
plt.title('P1')
plt.imshow( pre_grande4[5] , cmap = 'gray', vmin = 80, vmax = 700)

pre_list = [3, 5, 5, 5]

#%%
a = np.mean(post)/np.mean(pre)

plt.figure()
plt.title('pre')
plt.imshow( pre*a , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('post')
plt.imshow( post , cmap = 'gray', vmin = 80, vmax = 700)

#%%
post = post_grande[ 5 ]
pre, ZYX = correct_driff_3D( pre_grande1, post, 300, info = True )
c = cel_pre1
mascara = np.zeros( [1219]*2 )
mascara[ 100 + ZYX[1] : 1124  + ZYX[1] , -1124  + ZYX[2] : -100 + ZYX[2]  ] = m1
N = 1

m1_grande = np.copy(mascara)
#%%
celula_pre = np.ones( [1219]*2 )*np.mean(c)
celula_pre[ 100 + ZYX[1] : 1124  + ZYX[1] , -1124  + ZYX[2] : -100 + ZYX[2]  ] = c

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

inf = 120
a = np.mean(post)/np.mean(pre)
pre_plot = np.copy( (pre+5)*a - inf )
post_plot = np.copy(post - inf )
pre_plot[ pre < 0 ] = 0
pre_plot[ post < 0 ] = 0

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
plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 200, alpha = 1)
plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 200, alpha = 0.5)
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
vecinos = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]

mascara0 = np.zeros(mascara.shape)
img_s = smooth( mascara, 3 )
mascara0[ img_s > 0.5 ] = 1



y_borde = [600]
x_borde = []
j = 0
while len(x_borde) == 0:
    if mascara0[600,j] == 1:
        x_borde.append(j-1)
    j += 1    


# while (y_borde[-1] - y_borde[0])**2 + (x_borde[-1] - x_borde[0])**2 <= 1 and len( x_borde ) > 1000:
seguir = True
while seguir:
    x0 = x_borde[-1] 
    y0 = y_borde[-1]
    for j in range(8):
        v0 = mascara0[ y0 + vecinos[j-1][0], x0 + vecinos[j-1][1] ]
        v1 = mascara0[   y0 + vecinos[j][0],   x0 + vecinos[j][1] ]
        if v0 == 0 and v1 == 1:
            x_borde.append( x0 + vecinos[j-1][1] )
            y_borde.append( y0 + vecinos[j-1][0] )
    if ( x_borde[-1] == x_borde[0] and y_borde[-1] == y_borde[0] and len(x_borde) > 1 ) or len(x_borde) > 10000:
        seguir = False


borde4 = np.concatenate( (  np.reshape( np.array( y_borde ), [1, len(y_borde)]) , np.reshape( np.array( x_borde ) ,  [1, len(y_borde)] ) ) , axis = 0 )


plt.imshow( mascara0, cmap='gray' , alpha = 0.1)
plt.plot(borde4[1],borde4[0],c = 'b', linewidth = 2 )

#%%
plt.figure()
plt.imshow( m1_grande, cmap='gray' , alpha = 0.01)
plt.plot(borde1[1],borde1[0],c = (0.298, 0.447, 0.690), linewidth = 2, label = "T1" )
plt.plot(borde2[1],borde2[0],c = (0.866, 0.517, 0.321), linewidth = 2, label = "T2" )
plt.plot(borde3[1],borde3[0],c = (0.333, 0.658, 0.407), linewidth = 2, label = "T3" )
plt.plot(borde4[1],borde4[0],c = (0.768, 0.305, 0.321), linewidth = 2, label = "T4" )
plt.legend()


plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 35, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")



#%%












