# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:26:49 2023

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
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = "Times New Roman"

cm_crimson = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255)) for i in range(1000)] )
cm_green = ListedColormap( [(0,128*i/(999*255),0) for i in range(1000)] )
cm_yellow = ListedColormap( [( (220*i/(999*255)),128*i/(999*255),0) for i in range(1000)] )
cm_y = ListedColormap( [(1, 1, 1), (1, 1, 0)] )   # Blanco - Amarillo

c0 = (0.122, 0.467, 0.706)
c1 = (1.000, 0.498, 0.055)
c2 = (0.173, 0.627, 0.173)
c3 = (0.839, 0.152, 0.157)

cm1 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm2 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm4 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )

#MCF7 D30 R4 cel9 del 1/9   0
#MCF7 C30 R5 cel5 del 1/9   1
#MCF10 D04 R9 5/10          2
#MCF10 G18 R25 del 19/10    3
#%% Invocacion

# cel = 0
path = r"C:\Users\gonza\1\Tesis\2023\\"
nombres = [ 'MCF7 D30_R04', 'MCF7 C30_R05', 'MCF10 D04_R09', 'MCF10 G18_R25'  ]
regiones = [ 4, 5, 9, 25 ]
img_trans = [ 0, 0, 2, 2 ]
As = [ 0.85, 0.8, 0.8, 0.75]
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

b = border(mascara, 600)


#%%
cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')
print(ps)

#%%


















#%% Dependencia Axial de la deformación
z0 = 2

it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = 8#int(np.round(vi/2**(it-1)/3)) 

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
suave0 = 3
fs = 'x-large'

# l = int( int( 1024//vi + 1 )*4 )
zf = min(len(stack_pre),len(stack_post))-1

pre = stack_pre[5]
post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )
delta_z = ZYX[0] - 5

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
x, y = dominio

l = len(x)

Yz = np.zeros([zf-z0, l, l])
Xz = np.zeros([zf-z0, l, l])

Yz_s = np.zeros([zf-z0, l, l ])
Xz_s = np.zeros([zf-z0, l, l ])

mascara_grosa = np.zeros( [l]*2 )
for j in range(l):
    for i in range(l):
        if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
            mascara_grosa[i,j] = 1

# mascara_grosa3 = mascara_grosa
for z in range(z0,zf,1):
    print(z)
    pre = stack_pre[z]
    post = correct_driff( stack_post[z+delta_z], pre, 20 )

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio

    Xz_s[z-z0] = X_s*mascara_grosa
    Yz_s[z-z0] = Y_s*mascara_grosa

    Xz[z-z0] = X_nmt*mascara_grosa
    Yz[z-z0] = Y_nmt*mascara_grosa
    
    scale0 = 100
    plt.figure()
    plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
    # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'Z = ' + str(z) )
    plt.xlim([0,1023])
    plt.ylim([1023,0])
    plt.show()