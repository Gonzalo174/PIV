# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:33:24 2023

@author: Usuario
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
cm0 = ListedColormap(c0)
c3 = [(1, 1, 1), (1, 1, 0)]
cm3 = ListedColormap(c3)

c1 = []
c2 = []
for i in range(1000):
    c1.append((i/999,0,0))
    c2.append((0,i/999,0))

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)


#%% Import
# path = r"C:\Users\gonza\+1\Tesis\2023\\"
path = r"D:\Gonzalo\\"
carpetas = ["23.10.05 - gon MCF10 1 - A04", "23.10.05 - gon MCF10 2 - D04", "23.10.05 - gon MCF10 3 - E04", "23.10.06 - gon MCF10 4 - C04", "23.10.19 - gon MCF10 6 - G18", "23.10.20 - gon MCF10 7 - I18" ]
distribucion = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5 ]
pre10 =  [ 8, 4, 5, 7, 4, 4, 4, 4, 4, 4, 5, 4, 5, 2, 4, 4, 4, 4, 4,  4, 6, 5, 6, 5, 5,  3, 4, 5, 5, 3 ]
post10 = [ 8, 4, 6, 6, 2, 5, 3, 3, 4, 2, 5, 4, 4, 4, 4, 5, 5, 4, 4,  4, 5, 7, 8, 4, 6,  4, 5, 6, 4, 4 ]


cs = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 22, 25 ] 
ss = [14, 15, 17, 18, 26, 27, 28, 29]

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


for r in ss:
    full_path1 = path + carpetas[ distribucion[r-1] ]

    name = carpetas[ distribucion[r-1] ][-3:] + "_R" + str(int( 100 + r ))[1:]
    print(name)
    metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == r ]

    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution

    stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
    celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
    celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,2+pre10[r-1]-post10[r-1]]
    mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_00um.csv")
    
    pre = stack_pre[ pre10[r-1] ]
    post = correct_driff( stack_post[ post10[r-1] ], pre, 50 )
    if r == 17:
        pre[950:,900:] = stack_pre[ pre10[r-1]+6, 950:,900:]
        post[950:,900:] = stack_pre[ pre10[r-1]+6, 950:,900:]
        
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


    mascara10 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_10um.csv")
    if mascara10[1,1] == 1:
        mascara10 = 1 - mascara10    
    mascara20 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_20um.csv")
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



data.to_csv( "dataMCF10SS.csv" )





























#%% PIV + NMT + Suavizado


pre10[0] = 8
post10[0] = 8

r = 1
full_path1 = path + carpetas[ distribucion[r-1] ]

name = carpetas[ distribucion[r-1] ][-3:] + "_R" + str(int( 100 + r ))[1:]
print( name ) 

metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == r ]

field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,2+pre10[r-1]-post10[r-1]]
mascara = np.loadtxt( path + r"PIV\Mascaras MCF10\\" + name + "_m_00um.csv")

pre = stack_pre[ pre10[r-1] ]
post = correct_driff( stack_post[ post10[r-1] ], pre, 50 )
   
# if r == 1:
#     delta = 4
#     pre_profundo = stack_pre[ pre10[r+1] + delta ]
#     post_profundo = correct_driff( stack_post[ post10[r+1] + delta ], pre_profundo, 50 )
    
#     sat = np.zeros([1024]*2)
#     sat[ pre > 1000 ] = 1
#     sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
#     plt.imshow(sat)
#     pre = pre*(1-sat) + pre_profundo*sat
#     post = post*(1-sat) + post_profundo*sat


vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Smooth3"
# modo = "No control"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
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

c0 = [(0, 0, 0), (0, 0, 0)]
cm0 = ListedColormap(c0)

c1 = []
c2 = []
for i in range(1000):
    c1.append((i/999,0,0))
    c2.append((0,i/999,0))

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)

scale0 = 100
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/pixel_size
scale_unit = 'µm'  # Unit of the scale bar

wind = vi/( 2**(it-1) )
d = int( ( resolution - len(Y_nmt)*wind )/2   )

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + 50  # Starting x-coordinate of the scale bar
start_y = resolution -( 2*wind ) + 10# Starting y-coordinate of the scale bar



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
plt.imshow( mascara, cmap = "Reds", alpha = 0.4 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

# plt.savefig(name + '_figura.png')
plt.show()


# c_pre_a = np.reshape( celula, (1, 1, *pre.shape)  )
# c_post_a = np.reshape( celula_post, (1, 1, *pre.shape)  )

# pre_a = np.reshape( pre, (1, 1, *pre.shape)  )
# post_a = np.reshape( post, (1, 1, *pre.shape)  ).astype(dtype=np.uint16)


# archivo1 = np.concatenate( (c_pre_a,c_post_a), axis = 1 )
# archivo2 = np.concatenate( (pre_a,post_a) , axis = 1 )

# archivo0 = np.concatenate( (archivo1,archivo2), axis = 0 )







