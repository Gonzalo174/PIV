# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:23:33 2023

@author: gonza
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp 
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import imageio.v3 as iio
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import oiffile as of
import os

#%%
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = "Arial"

cm_crimson = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255)) for i in range(1000)] )
cm_green = ListedColormap( [(0,128*i/(999*255),0) for i in range(1000)] )
cm_yellow = ListedColormap( [( (220*i/(999*255)),128*i/(999*255),0) for i in range(1000)] )
cm_y = ListedColormap( [(1, 1, 1), (1, 1, 0)] )   # Blanco - Amarillo
cm_ar = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (0.839, 0.152, 0.157)] ) 
cm_aa = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (1.000, 0.498, 0.055)] ) 
cm_aa2 = ListedColormap( [(0.122, 0.467, 0.706), (0, 0, 0), (1.000, 0.498, 0.055)] ) 

c0 = (0.122, 0.467, 0.706)
c1 = (1.000, 0.498, 0.055)
c2 = (0.173, 0.627, 0.173)
c3 = (0.839, 0.152, 0.157)
colores = [c0, c1, c2, c3]

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )
color_maps = [cm0, cm1, cm2, cm3]

# 7cs 	(11,4)
# 7ss	(7,2)
# 10cs	25
# 10ss	27
#%%
cel = 0

#%% Invocacion
path = r"C:\Users\gonza\1\Tesis\2023\\"
nombres = [ 'MCF7 D30_R04', 'MCF7 C23_R02', 'MCF10 G18_R25', 'MCF10 I18_R27' ]
regiones = [ 4, 2, 25, 27 ]
img_trans = [ 0, 0, 2, 2 ]
As = [ 0.85, 0.8, 0.8, 0.75]
ps_list = [0.0804, 0.0918, 0.1007, 0.1007]

carpetas = [ "23.09.01 - gon MCF7 11 - D30", "23.08.25 - gon MCF7 7 - C23", "23.10.19 - gon MCF10 6 - G18", "23.10.20 - gon MCF10 7 - I18" ]
full_path = path + carpetas[ cel ]

print(nombres[cel])
metadata = pd.read_csv( full_path + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == regiones[cel] ]
zoom = metadata_region["Zoom"].values[0]
ps = 1/(4.97*zoom)

stack_pre = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
celula_pre = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,img_trans[cel]]

if cel == 0 or cel == 1:
    mascara =  1 - np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_00um.png")
    mascara10 =  1 - np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_10um.png")
    mascara20 =  1 - np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_20um.png")

elif cel == 2 or cel ==3:
    mascara =  np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_00um.csv")
    mascara10 =  np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_10um.csv")
    mascara20 =  np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_20um.csv")

if cel == 0:
    mascara = iio.imread(r'C:\Users\gonza\1\Tesis\Poster SAB\mascara0.png')

b = border(mascara, 600, 5)

#%%
cel = 2
runcell('Invocacion, #1', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

#%%






#%%

celulas = []
bordes = []
mas00 = []
mas10 = []
listaX = []
listaY = []
listax = []
listay = []

for iterador in range(4):
    cel = iterador
    runcell('Invocacion, #1', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

    pre = stack_pre[5]
    post, data = correct_driff_3D( stack_post, pre, 50, info = True )

    # Determinación del campo de deformación
    it = 3
    vi = int( int( 3/ps )*2**(it-1) )
    bordes_extra = 8
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 2.5
    modo = "Smooth3"
    suave0 = 3

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio

    fondo = median_blur(celula_pre, 50)
    celulas.append( celula_pre - fondo )
    bordes.append( b )
    mas00.append( mascara )
    mas10.append( mascara10 )
    listaX.append( X_s )
    listaY.append( Y_s )
    listax.append( x )
    listay.append( y )

#%%
esc = 20
separacion = 1.5
fs = 'x-small'
plt.figure( figsize = [5, 11] )
plt.subplots_adjust(wspace=0.04, hspace=0.0005) 
cel = 2
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel]

plt.subplot(4,2,1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed', lw = 0.75  )
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,2,2)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( esc, pixel_size = ps_list[cel], sep = separacion, img_len = 990,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])


cel = 3
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel]

plt.subplot(4,2,3)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed', lw = 0.75    )
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,2,4)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990, sep = separacion,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])



cel = 0
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel]

plt.subplot(4,2,5)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed', lw = 0.75    )
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,2,6)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990, sep = separacion,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])



cel = 1
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel]


plt.subplot(4,2,7)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed' , lw = 0.75   )
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,2,8)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990, sep = separacion,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])















#%%
plt.figure(figsize = [11, 11], tight_layout=True)
cel = 0
runcell('Invocacion, #1', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])


cel = 1
runcell('Invocacion, #1', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])



cel = 2
runcell('Invocacion, #1', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])



cel = 3
runcell('Invocacion, #1', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])


















































#MCF7 D30 R4 cel9 del 1/9   0
#MCF7 C30 R5 cel5 del 1/9   1
#MCF10 D04 R9 5/10          2
#MCF10 G18 R25 del 19/10    3
#%%
cel = 0

#%% Invocacion
path = r"C:\Users\gonza\1\Tesis\2023\\"
nombres = [ 'MCF7 D30_R04', 'MCF7 C30_R05', 'MCF10 D04_R09', 'MCF10 G18_R25'  ]
regiones = [ 4, 5, 9, 25 ]
img_trans = [ 0, 0, 2, 2 ]
As = [ 0.85, 0.8, 0.8, 0.75]
ps_list = [0.0804, 0.0918, 0.1007, 0.1007]

carpetas = [ "23.09.01 - gon MCF7 11 - D30", "23.09.01 - gon MCF7 10 - C30",  "23.10.05 - gon MCF10 2 - D04", "23.10.19 - gon MCF10 6 - G18"  ]
full_path = path + carpetas[ cel ]

print(nombres[cel])
metadata = pd.read_csv( full_path + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == regiones[cel] ]
zoom = metadata_region["Zoom"].values[0]
ps = 1/(4.97*zoom)

stack_pre = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
celula_pre = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,img_trans[cel]]

if cel == 0 or cel == 1:
    mascara =  1 - np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_00um.png")
    mascara10 =  1 - np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_10um.png")
    mascara20 =  1 - np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_20um.png")

elif cel == 2 or cel ==3:
    mascara =  np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_00um.csv")
    mascara10 =  np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_10um.csv")
    mascara20 =  np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_20um.csv")

if cel == 0:
    mascara = iio.imread(r'C:\Users\gonza\1\Tesis\Poster SAB\mascara0.png')

b = border(mascara, 600)

#%%
cel = 0
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Poster SAB.py')

#%%

celula_post = of.imread( full_path + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,img_trans[cel]]
fondo_pre = median_blur( celula_pre, 50 )
fondo_post = median_blur( celula_post, 50 )

#%%
fs = 'medium'
fs1 = 'small'
plt.figure(figsize = [7,7])#,tight_layout=True)
a = np.mean( stack_pre[5] )/np.mean( stack_post[5] )

plt.subplots_adjust(wspace=0.001, hspace=0.03)
plt.subplot(2,2,1)
plt.imshow(celula_pre - fondo_pre, cmap = "gray")
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
barra_de_escala( 20, pixel_size = ps, sep = 2,  font_size = fs, color = 'k', more_text = 'PRE' )

plt.subplot(2,2,2)
plt.imshow(celula_post - fondo_post, cmap = "gray")
circle = plt.Circle((410, 495), 130, color='w', ls = 'dashed', lw = 0.75 , fill=False)
plt.gca().add_patch(circle)
barra_de_escala( 20, pixel_size = ps, sep = 2,  font_size = fs, color = 'k', more_text = 'POST', text = False )

plt.subplot(2,2,3)
plt.imshow( stack_pre[5], cmap = cm_crimson, vmin = 150, vmax = 500 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
barra_de_escala( 20, pixel_size = ps, sep = 2,  font_size = fs, color = 'w', text = False)
plt.text(512, 40, 'DEFORMED SUBSTRATE', color='w', weight='bold', ha='center', va = 'top', fontsize = fs1 )

plt.subplot(2,2,4)
plt.imshow( stack_post[5], cmap = cm_crimson, vmin = 150, vmax = 500/(a + 0.1) )
# plt.plot( b[1], b[0], c = 'w' )
barra_de_escala( 20, pixel_size = ps, sep = 2,  font_size = fs, color = 'w', text = False )
plt.text(512, 40, 'RELAXED SUBSTRATE', color='w', weight='bold', ha='center', va = 'top', fontsize = fs1 )

plt.show()

#%%
pre = stack_pre[4]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = 8

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[0])
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

#%%
fs = 'small'

plt.figure(figsize = [7,7])#, tight_layout = True)
a = np.mean( pre )/np.mean( post )
a_x, b_x = 580, 880
a_y, b_y = 370, 670

xo, yo = 860, 660

plt.subplots_adjust(wspace=0.03, hspace=0.03)
plt.subplot(1,3,1)
plt.imshow(celula_pre - fondo_pre, cmap = "gray")
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
# barra_de_escala( 10, pixel_size = ps, img_len = np.abs(a_x - b_x), sep = 0.8,  font_size = fs, color = 'k' )
plt.xlim([a_x,b_x])
plt.ylim([b_y,a_y])
plt.xticks([])
plt.yticks([])
for i in np.arange(-5, -10, -0.1):
    plt.plot([ xo-124, xo ], [yo + i, yo + i], color='k', linewidth = 2)
plt.text( xo - 62, yo - 20 , '10 µm', color= 'k', weight='bold', ha='center', va = 'bottom', fontsize = fs )


plt.subplot(1,3,2)
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 400 )
plt.imshow( post, cmap = cm_green, vmin = 150, vmax = 400/(a + 0.1), alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
# barra_de_escala( 10, pixel_size = ps, img_len = np.abs(a_x - b_x), sep = 0.8,  font_size = fs, color = 'w', text = False)
plt.xlim([a_x,b_x])
plt.ylim([b_y,a_y])
plt.xticks([])
plt.yticks([])
for i in np.arange(-5, -10, -0.1):
    plt.plot([ xo-124, xo ], [yo + i, yo + i], color='w', linewidth = 2)

plt.subplot(1,3,3)
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.9 )
plt.quiver(x,y,X_s,-Y_s, width = 0.006, scale = 100,  pivot='tail')
# barra_de_escala( 10, sep = 0.8, img_len = np.abs(a_x - b_x),  font_size = fs, color = 'k', text = False)
plt.xlim([a_x,b_x])
plt.ylim([b_y,a_y])
plt.xticks([])
plt.yticks([])
for i in np.arange(-5, -10, -0.1):
    plt.plot([ xo-124, xo ], [yo + i, yo + i], color='k', linewidth = 2)

plt.show()













