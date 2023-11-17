# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:59:54 2023

@author: gonza
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
plt.rcParams['figure.figsize'] = [7,7]
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = "Times New Roman"

cm_crimson = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255)) for i in range(1000)] )
cm_orange = ListedColormap( [(i/(999),165*i/(999*255),0) for i in range(1000)] )
cm_green = ListedColormap( [(0,128*i/(999*255),0) for i in range(1000)] )
cm_yellow = ListedColormap( [( (220*i/(999*255)),128*i/(999*255),0) for i in range(1000)] )
cm_y = ListedColormap( [(1, 1, 1), (1, 1, 0)] )   # Blanco - Amarillo
cm_blanco = ListedColormap( [(0, 0, 0), (1, 1, 1)] )   # Blanco - Amarillo

 
cm1 = ListedColormap( [(1, 1, 1), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765) ] )
cm2 = ListedColormap( [(1, 1, 1), (1.0, 0.4980392156862745, 0.054901960784313725)] )
cm3 = ListedColormap( [(1, 1, 1), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)] )
cm4 = ListedColormap( [(1, 1, 1), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)] )
cm5 = ListedColormap( [(1, 1, 1), (0.5803921568627451, 0.403921568627451, 0.7411764705882353)] )
#%%
import seaborn as sns

# Obtiene la paleta de colores por defecto de seaborn
default_palette = sns.color_palette()

# Muestra los códigos hexadecimales de los primeros 5 colores
for i, color in enumerate(default_palette[:5]):
    print(color)


#%% Invocacion

path = r"C:\Users\gonza\1\Tesis\2023\\"
As = [ 0.5, 0.8 ]
ps = 105.47/1600
carpetas = [ "23.05.12 - Celulas en geles B4", "23.05.18 - Celulas en geles OA Orange" ]
full_pathC = path + carpetas[0]
full_pathO = path + carpetas[1]

stack_pre_C = of.imread( full_pathC + r"\\B1-R3-07-60X-pw20-k2-pre.oif" )[0]
stack_post_C = of.imread( full_pathC + r"\\B1-R3-14-60X-pw20-k2-post.oif" )[0]
celula_pre_C = np.fliplr(of.imread( full_pathC + r"\\B1-R3-06-60X-pw0.5-k0-tra.oif" )[1,1])

stack_pre_O = of.imread( full_pathO + r"\\02-BOA-R1-60X-pw0.2-k0-pre.oif" )[0]
stack_post_O = of.imread( full_pathO + r"\\06-BOA-R1-60X-pw0.2-k2-post.oif" )[0]
celula_pre_O = of.imread( full_pathO + r"\\02-BOA-R1-60X-pw0.2-k0-pre.oif" )[1,0]


mascara_C1 =  np.fliplr(1 - plt.imread( full_pathC + r"\\celula_Crimson_R3_cell1.png" )[:,:,1])
mascara_C2 =  np.fliplr(1 - plt.imread( full_pathC + r"\\celula_Crimson_R3_cell2.png" )[:,:,1])
mascara_C3 =  np.fliplr(1 - plt.imread( full_pathC + r"\\celula_Crimson_R3_cell3.png" )[:,:,1])
mascara_C4 =  np.fliplr(1 - plt.imread( full_pathC + r"\\celula_Crimson_R3_cell4.png" )[:,:,2])
mascara_C5 =  np.fliplr(1 - plt.imread( full_pathC + r"\\celula_Crimson_R3_cell5.png" )[:,:,2])

mascara_O1 =  plt.imread( full_pathO + r"\\CelO_1.png" )[:,:,0]
mascara_O2 =  plt.imread( full_pathO + r"\\CelO_2.png" )[:,:,0]

borde_C10 = contour( mascara_C1 )
borde_C20 = contour( mascara_C2 )
borde_C30 = contour( mascara_C3 )
borde_C40 = contour( mascara_C4 )
borde_C50 = contour( mascara_C5 )

borde_O10 = contour( mascara_O1 )
borde_O20 = contour( mascara_O2 )

borde_C1 = []
borde_C2 = []
borde_C3 = []
borde_C4 = []
borde_C5 = []

borde_O1 = []
borde_O2 = []

for j in range(1600):
    for i in range(1600):
        if borde_C10[j,i] != 0:
            borde_C1.append( [j,i] )
        if borde_C20[j,i] != 0:
            borde_C2.append( [j,i] )
        if borde_C30[j,i] != 0:
            borde_C3.append( [j,i] )
        if borde_C40[j,i] != 0:
            borde_C4.append( [j,i] )
        if borde_C50[j,i] != 0:
            borde_C5.append( [j,i] )
        if borde_O10[j,i] != 0:
            borde_O1.append( [j,i] )
        if borde_O20[j,i] != 0:
            borde_O2.append( [j,i] )

borde_C1 = np.array(borde_C1)
borde_C2 = np.array(borde_C2)
borde_C3 = np.array(borde_C3)
borde_C4 = np.array(borde_C4)
borde_C5 = np.array(borde_C5)

borde_O1 = np.array(borde_O1)
borde_O2 = np.array(borde_O2)

#%%

cel = "O"

if cel == "C":
    stack_pre, stack_post, celula_pre = stack_pre_C, stack_post_C, celula_pre_C
    mascara1, mascara2, mascara3, mascara4, mascara5 = mascara_C1, mascara_C2, mascara_C3, mascara_C4, mascara_C5
    borde1, borde2, borde3, borde4, borde5 = borde_C1, borde_C2, borde_C3, borde_C4, borde_C5

if cel == "O":
    stack_pre, stack_post, celula_pre = stack_pre_O, stack_post_O, celula_pre_O
    mascara1, mascara2 = mascara_O1, mascara_O2
    borde1, borde2 = borde_O1, borde_O2

#%%

plt.figure( figsize=[12,4] )
desvios = desvio_por_plano(stack_pre)
desvios2 = desvio_por_plano(stack_post)
plt.plot(desvios, 'o', )
plt.plot(desvios2, 'o', )
plt.grid()
plt.xticks(np.arange(len(stack_pre)))
plt.show()

#%%
pre = stack_pre[5]
# post = correct_driff( np.fliplr( stack_post[5] ) , pre,  50  )
post, data = correct_driff_3D( stack_post, pre, 50, info = True )


VM = 600
if cel == 'C':
    VM = 300
Vm = 100
fs = 'xx-large'
cm = cm_orange
alpha = np.mean( pre )/np.mean( post )

plt.figure(figsize = [11, 11], tight_layout=True)

plt.subplot(2,2,1)
plt.imshow( pre, cmap = cm, vmin = Vm, vmax = VM )
barra_de_escala( 20, pixel_size = ps, img_len = 1600, sep = 3,  font_size = fs )

plt.subplot(2,2,2)
plt.imshow( post, cmap = cm, vmin = Vm, vmax = VM/alpha )
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 3,  font_size = fs )

plt.subplot(2,2,3)
plt.imshow( pre, cmap = cm_orange, vmin = Vm, vmax = VM - 50  )
plt.imshow( post, cmap = cm_green, vmin = Vm, vmax = VM/alpha - 50, alpha = 0.5 )
plt.scatter( borde1[:,1] ,borde1[:,0], c = 'w', marker = '.', s = 0.25 )
plt.scatter( borde2[:,1] ,borde2[:,0], c = 'w', marker = '.', s = 0.25 )
if cel == 'C':
    plt.scatter( borde3[:,1] ,borde3[:,0], c = 'w', marker = '.', s = 0.25 )
    plt.scatter( borde4[:,1] ,borde4[:,0], c = 'w', marker = '.', s = 0.25 )
    plt.scatter( borde5[:,1] ,borde5[:,0], c = 'w', marker = '.', s = 0.25 )
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 3,  font_size = fs )

plt.subplot(2,2,4)
plt.imshow( celula_pre, cmap = 'gray' )
plt.scatter( borde1[:,1] ,borde1[:,0], c = 'w', marker = '.', s = 0.25 )
plt.scatter( borde2[:,1] ,borde2[:,0], c = 'w', marker = '.', s = 0.25 )
if cel == 'C':
    plt.scatter( borde3[:,1] ,borde3[:,0], c = 'w', marker = '.', s = 0.25 )
    plt.scatter( borde4[:,1] ,borde4[:,0], c = 'w', marker = '.', s = 0.25 )
    plt.scatter( borde5[:,1] ,borde5[:,0], c = 'w', marker = '.', s = 0.25)
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 3,  font_size = fs, color = 'k' )


plt.show()

print(data)

#%%
A = 0.75
desvios_bin, limit = busca_esferas( pre, ps = ps, th = A )
plt.figure( figsize = [8, 8] )
plt.imshow( pre[ :limit , :limit ], cmap = cm_orange, vmin = 150, vmax = 300 )
plt.imshow( desvios_bin, cmap = 'gray', alpha = 0.09, extent = [0,limit,limit,0])
barra_de_escala( 20, pixel_size = ps, img_len = 1570,  sep = 2.5, more_text = 'Célula ' + str(cel), a_lot_of_text = str( int( np.mean( desvios_bin )*100 ) ) + '%', font_size = fs )

# %%
it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = int(np.round(vi/2**(it-1))/4) 

# j_p, i_p = np.random.randint( len(x) - 2 ) + 1, np.random.randint( len(x) - 2 ) + 1
# control0 = [(j_p,i_p)]
control0 = [(0,0)]
# control0 = [(-1,-1)]

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
# modo = "Fit"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = A, control = control0)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

#%%
scale0 = 100
plt.imshow( mascara1, cmap = cm1, alpha = 1 )
plt.imshow( mascara2, cmap = cm2, alpha = 0.3 )
if cel == 'C':
    plt.imshow( mascara3, cmap = cm3, alpha = 0.3 )
    plt.imshow( mascara4, cmap = cm4, alpha = 0.3 )
    plt.imshow( mascara5, cmap = cm5, alpha = 0.3 )
plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 2,  font_size = fs, color = 'k' )
plt.xlim([0,1600])
plt.ylim([1600,0])

print( np.round( np.max( np.sqrt(X_nmt**2 + Y_nmt**2) )*ps, 2)  )


#%%
scale0 = 100
plt.imshow( mascara1, cmap = cm1, alpha = 1 )
plt.imshow( mascara2, cmap = cm2, alpha = 0.3 )
if cel == 'C':
    plt.imshow( mascara3, cmap = cm3, alpha = 0.3 )
    plt.imshow( mascara4, cmap = cm4, alpha = 0.3 )
    plt.imshow( mascara5, cmap = cm5, alpha = 0.3 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 2,  font_size = fs, color = 'k' )
plt.xlim([0,1600])
plt.ylim([1600,0])

print( np.round(np.max( np.sqrt(X_s**2 + Y_s**2) )*ps, 2)  )


#%%

fs = 'large'
plt.figure( figsize = [11, 22], tight_layout=True )
plt.subplot(1,3,1)
plt.imshow( celula_pre, cmap = 'gray' )
plt.scatter( borde1[:,1] ,borde1[:,0], c = 'w', marker = '.', s = 0.15 )
plt.scatter( borde2[:,1] ,borde2[:,0], c = 'w', marker = '.', s = 0.15 )
if cel == 'C':
    plt.scatter( borde3[:,1] ,borde3[:,0], c = 'w', marker = '.', s = 0.15 )
    plt.scatter( borde4[:,1] ,borde4[:,0], c = 'w', marker = '.', s = 0.15 )
    plt.scatter( borde5[:,1] ,borde5[:,0], c = 'w', marker = '.', s = 0.15)
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 3.5,  font_size = fs, color = 'k' )


plt.subplot(1,3,2)
plt.imshow( pre, cmap = cm_orange, vmin = Vm, vmax = VM - 50  )
plt.imshow( post, cmap = cm_green, vmin = Vm, vmax = VM/alpha - 50, alpha = 0.5 )
plt.scatter( borde1[:,1] ,borde1[:,0], c = 'w', marker = '.', s = 0.15 )
plt.scatter( borde2[:,1] ,borde2[:,0], c = 'w', marker = '.', s = 0.15 )
if cel == 'C':
    plt.scatter( borde3[:,1] ,borde3[:,0], c = 'w', marker = '.', s = 0.15 )
    plt.scatter( borde4[:,1] ,borde4[:,0], c = 'w', marker = '.', s = 0.15 )
    plt.scatter( borde5[:,1] ,borde5[:,0], c = 'w', marker = '.', s = 0.15 )
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 3.5,  font_size = fs )


scale0 = 100
plt.subplot(1,3,3)
plt.imshow( mascara1, cmap = cm1, alpha = 1 )
plt.imshow( mascara2, cmap = cm2, alpha = 0.3 )
if cel == 'C':
    plt.imshow( mascara3, cmap = cm3, alpha = 0.3 )
    plt.imshow( mascara4, cmap = cm4, alpha = 0.3 )
    plt.imshow( mascara5, cmap = cm5, alpha = 0.3 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 20, pixel_size = ps, img_len = 1600,  sep = 3.5,  font_size = fs, color = 'k' )
plt.xlim([0,1600])
plt.ylim([1600,0])


plt.show()















