# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:25:05 2023

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

c0 = [(0, 0, 0), (0, 0, 0)]
c3 = [(1, 1, 1), (1, 1, 0)]
cm0 = ListedColormap(c0)
cm3 = ListedColormap(c3)

c1 = []
c2 = []
for i in range(1000):
    c1.append((i/999,0,0))
    c2.append((0,i/999,0))

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)

#%% Import
path = r"C:\Users\gonza\1\Tesis\2023\\"
# path = r"D:\Gonzalo\\"
carpetas = ["23.08.17 - gon MCF7 1 - C16", "23.08.18 - gon MCF7 2 - B16", "23.08.18 - gon MCF7 3 - A16", "23.08.24 - gon MCF7 4 - A23", "23.08.24 - gon MCF7 5 - B23", "23.08.25 - gon MCF7 6 - D23", "23.08.25 - gon MCF7 7 - C23", "23.08.31 - gon MCF7 8 - B30", "23.08.31 - gon MCF7 9 - A30", "23.09.01 - gon MCF7 10 - C30", "23.09.01 - gon MCF7 11 - D30"]

cs =  [ (11,2), (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
pre_post7 = {(11,2):(6,5), (11,3):(4,3), (11,4):(4,4), (10,1):(4,3), (10,2):(8,4), (10,5):(4,4), (9,1):(3,3), (1,1):(8,7),    (8,2):(4,4), (8,3):(5,5), (7,1):(6,4), (7,2):(5,4), (6,2):(6,5), (6,3):(5,4), (6,4):(4,5), (5,4):(3,4), (4,1):(7,7), (3,3):(6,6)   }

#%%

cel = []
defo00 = []
Ddefo00 = []
Mdefo00 = []
defo10 = []
Ddefo10 = []
Mdefo10 = []
defo20 = []
Ddefo20 = []
Mdefo20 = []
area = []


for tup in ss:
    print(tup)
    full_path1 = path + carpetas[ tup[0] - 1 ]

    name = carpetas[ tup[0] - 1 ][-3:] + "_R0" + str( tup[1] )
    print(name)
    metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == tup[1] ]    
    
    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution

    stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
    celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
    celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[1, 2 + pre_post7[tup][0] - pre_post7[tup][1] ]
    mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_00um.png")
    if mascara[1,1] == 1:
        mascara = 1 - mascara
    
    pre = stack_pre[ pre_post7[tup][0] ]
    if tup == (8,3):  
        post = correct_driff( stack_post[ pre_post7[tup][1] ], pre, 300 )
    elif tup == (7,1):  
        post = unrotate( stack_post[ pre_post7[tup][1] ], pre, 50, exploration_angle = 1)
    elif tup == (4,1):
        pre = np.concatenate( ( pre, np.ones([4, 1024])*np.mean(pre) ), axis = 0  )
        post = correct_driff( stack_post[ pre_post7[tup][1] ], pre, 50 )
    elif tup == (3,3):  
        post = unrotate( stack_post[ pre_post7[tup][1] ], pre, 50, exploration_angle = 1)    
    else:
        post = correct_driff( stack_post[ pre_post7[tup][1] ], pre, 50 )

    vi = 128
    it = 3
    bordes_extra = 10 # px

    Noise_for_NMT = 0.2
    Threshold_for_NMT = 5
    modo = "Smooth3"
    mapas = False
    suave0 = 3

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio


    inf = 120
    a = np.mean(post)/np.mean(pre)
    pre_plot = np.copy( (pre+5)*a - inf )
    post_plot = np.copy(post - inf )
    pre_plot[ pre < 0 ] = 0
    pre_plot[ post < 0 ] = 0

    scale0 = 100
    scale_length = 10  # Length of the scale bar in pixels
    scale_pixels = scale_length/pixel_size
    scale_unit = 'µm'  # Unit of the scale bar

    wind = vi/( 2**(it-1) )
    d = int( ( resolution - len(Y_nmt)*wind )/2   )

    # Add the scale bar
    scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    start_x = d + 50  # Starting x-coordinate of the scale bar
    start_y = resolution - ( 2*wind ) + 10# Starting y-coordinate of the scale bar

    plt.figure(figsize=(20,20), tight_layout=True)

    plt.subplot(2,2,1)
    plt.imshow( celula_pre , cmap = 'gray' )
    plt.xticks([])
    plt.yticks([])
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
    plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")

    plt.subplot(2,2,2)
    plt.imshow( celula_post , cmap = 'gray' )
    plt.xticks([])
    plt.yticks([])
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

    plt.subplot(2,2,3)
    plt.imshow( np.zeros(pre.shape), cmap = cm0 )
    plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
    plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)
    plt.xticks([])
    plt.yticks([])
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)

    plt.subplot(2,2,4)
    plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
    plt.imshow( mascara, cmap = cm3, alpha = 0.6 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,resolution])
    plt.ylim([resolution,0])

    plt.show()


    mascara10 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_10um.png")
    if mascara10[1,1] == 1:
        mascara10 = 1 - mascara10    
    mascara20 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_20um.png")
    if mascara20[1,1] == 1:
        mascara20 = 1 - mascara20
    
    Y_work, X_work = Y_s, X_s
    Y_cell_00 = []
    X_cell_00 = []
    Y_cell_10 = []
    X_cell_10 = []
    Y_cell_20 = []
    X_cell_20 = []
        
    l = len(Y_work)
    ps = pixel_size
    for j in range(l):
        for i in range(l):
            if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara[ int(x[j,i]), int(y[j,i]) ]) == 1:
                Y_cell_00.append(Y_work[j,i])
                X_cell_00.append(X_work[j,i])
            if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                Y_cell_10.append(Y_work[j,i])
                X_cell_10.append(X_work[j,i]) 
            if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara20[ int(x[j,i]), int(y[j,i]) ]) == 1:
                Y_cell_20.append(Y_work[j,i])
                X_cell_20.append(X_work[j,i])


    cel.append( name )
    defo00.append( np.mean( np.sqrt( np.array(Y_cell_00)**2 + np.array(X_cell_00)**2 )*ps )  )
    Ddefo00.append( np.std( np.sqrt( np.array(Y_cell_00)**2 + np.array(X_cell_00)**2 )*ps )  )
    Mdefo00.append( np.max( np.sqrt( np.array(Y_cell_00)**2 + np.array(X_cell_00)**2 )*ps )  )
    
    defo10.append( np.mean( np.sqrt( np.array(Y_cell_10)**2 + np.array(X_cell_10)**2 )*ps )  )
    Ddefo10.append( np.std( np.sqrt( np.array(Y_cell_10)**2 + np.array(X_cell_10)**2 )*ps )  )
    Mdefo10.append( np.max( np.sqrt( np.array(Y_cell_10)**2 + np.array(X_cell_10)**2 )*ps )  )
    
    defo20.append( np.mean( np.sqrt( np.array(Y_cell_20)**2 + np.array(X_cell_20)**2 )*ps )  )
    Ddefo20.append( np.std( np.sqrt( np.array(Y_cell_20)**2 + np.array(X_cell_20)**2 )*ps )  )
    Mdefo20.append( np.max( np.sqrt( np.array(Y_cell_20)**2 + np.array(X_cell_20)**2 )*ps )  )
    
    area.append( np.sum( mascara )*ps**2 )




#%%

data = pd.DataFrame()


data["Celula"] = cel
data["Promedio (0 um)"] = defo00
data["Error (0 um)"] = Ddefo00
data["Maximo (0 um)"] = Mdefo00
data["Promedio (10 um)"] = defo10
data["Error (10 um)"] = Ddefo10
data["Maximo (10 um)"] = Mdefo10
data["Promedio (20 um)"] = defo20
data["Error (20 um)"] = Ddefo20
data["Maximo (20 um)"] = Mdefo20
data["Area"] = area
data["Clase"] = ["MCF7SS"]*len(cel )



data.to_csv( "dataMCF7SS.csv" )















#%%

c0 = [(0, 0, 0), (1, 1, 1)]
cm0 = ListedColormap(c0)

for tup in CS:
    name = carpetas[ tup[0] - 1 ][-3:] + "_R0" + str( tup[1] )
    mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_00um.png")
    if mascara[1,1] == 1:
        mascara = 1 - mascara
    plt.figure( )
    plt.title( name )
    plt.imshow( mascara, cmap=cm0 )



















