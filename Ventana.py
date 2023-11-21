# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:27:26 2023

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

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )

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
#%% Dependencia con la ventana de exploración de al deformación
z0 = 5

bordes_extra = 8
ventanas = np.arange( int(6/ps), int(2/ps),-1)  # en px

Rz3 = []              
Rz3_dist = []

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
suave0 = 3
fs = 'x-large'

pre = stack_pre[z0]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

df = pd.DataFrame()

for vs in ventanas:
    # Determinación del campo de deformación
    it = 3
    vi = int( int( vs )*2**(it-1) )

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio
    
    # Ajuste de las dimenciones de la mascara a las del campo
    Rz_dist = []
    
    l = len(X_s)
    mascara_grosa = np.zeros( [l]*2 )
    for j in range(l):
        for i in range(l):
            if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                mascara_grosa[i,j] = 1
                Rz_ij = np.sqrt((X_nmt[i,j])**2 + (Y_nmt[i,j])**2)*ps
                Rz_dist.append( Rz_ij )
    
    # Recopilación de datos
    Rz = np.sqrt((X_nmt)**2 + (Y_nmt)**2)*mascara_grosa*ps
    Rz3.append( np.sum(Rz)/np.sum(mascara_grosa) )
    Rz3_dist.append( Rz_dist )

    # Gráfico
    scale0 = 100
    plt.figure()
    plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
    # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'V = ' + str(np.round(vs,1)) )
    plt.xlim([0,1023])
    plt.ylim([1023,0])
    plt.show()