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

cm0 = ListedColormap( [(0, 0, 0), (0, 0, 0)] )               # Negro
cm1 = ListedColormap( [(i/999,0,0) for i in range(1000)] )   # Negro - Rojo
cm2 = ListedColormap( [(1,i/999,0) for i in range(1000)] )   # Negro - Verde
cm3 = ListedColormap( [(1, 1, 1), (1, 1, 0)] )               # Blanco - Amarillo

#%% Import
path = r"C:\Users\gonza\1\Tesis\2023\\"
# path = r"D:\Gonzalo\\"
carpetas = ["23.08.17 - gon MCF7 1 - C16", "23.08.18 - gon MCF7 2 - B16", "23.08.18 - gon MCF7 3 - A16", "23.08.24 - gon MCF7 4 - A23", "23.08.24 - gon MCF7 5 - B23", "23.08.25 - gon MCF7 6 - D23", "23.08.25 - gon MCF7 7 - C23", "23.08.31 - gon MCF7 8 - B30", "23.08.31 - gon MCF7 9 - A30", "23.09.01 - gon MCF7 10 - C30", "23.09.01 - gon MCF7 11 - D30"]

cs =  [ (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
pre_post7 = {(11,2):(6,5), (11,3):(4,3), (11,4):(4,4), (10,1):(4,3), (10,2):(8,4), (10,5):(4,4), (9,1):(3,3), (1,1):(6,5),    (8,2):(4,4), (8,3):(5,5), (7,1):(6,4), (7,2):(5,4), (6,2):(6,5), (6,3):(5,4), (6,4):(4,5), (5,4):(3,4), (4,1):(7,7), (3,3):(6,6)   }

#%%
cel = []

defo00 = []
Pdefo00 = []
DPdefo00 = []
Sdefo00 = []
DSdefo00 = []
Mdefo00 = []

defo10 = []
Pdefo10 = []
DPdefo10 = []
Sdefo10 = []
DSdefo10 = []
Mdefo10 = []

defo20 = []
Pdefo20 = []
DPdefo20 = []
Sdefo20 = []
DSdefo20 = []
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
    # pixel_size = field/resolution
    zoom = metadata_region["Zoom"].values[0]
    pixel_size = 1/(4.97*zoom)
    ps = pixel_size


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
        
    if tup == (1,1):
        delta = 3
        pre_profundo = stack_pre[ pre_post7[tup][0] + delta ]
        post_profundo = correct_driff( stack_post[ pre_post7[tup][1] + delta ], pre_profundo, 50 )
        
        sat = np.zeros([1024]*2)
        sat[ pre > 1000 ] = 1
        sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
        
        pre = pre*(1-sat) + pre_profundo*sat
        post = post*(1-sat) + post_profundo*sat

    # vi = 128
    vi = int( int( 3/ps )*4 )
    it = 3
    # bordes_extra = 10 # px
    bordes_extra = int(np.round(vi/9))

    Noise_for_NMT = 0.2
    Threshold_for_NMT = 2.5
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
    
    R_00 = np.sqrt( np.array(Y_cell_00)**2 + np.array(X_cell_00)**2 )*ps
    defo00.append( R_00  )
    Pdefo00.append( np.mean( R_00 ) )
    DPdefo00.append( np.std( R_00 ) )
    Sdefo00.append( np.sum( R_00 ) )
    DSdefo00.append( np.std( R_00 )*len(R_00)  )
    Mdefo00.append( np.max( R_00 )  )
    
    R_10 = np.sqrt( np.array(Y_cell_10)**2 + np.array(X_cell_10)**2 )*ps
    defo10.append( R_10  )
    Pdefo10.append( np.mean( R_10 ) )
    DPdefo10.append( np.std( R_10 ) )
    Sdefo10.append( np.sum( R_10 ) )
    DSdefo10.append( np.std( R_10 )*len(R_10)  )
    Mdefo10.append( np.max( R_10 )  )
        
    R_20 = np.sqrt( np.array(Y_cell_20)**2 + np.array(X_cell_20)**2 )*ps
    defo20.append( R_20  )
    Pdefo20.append( np.mean( R_20 ) )
    DPdefo20.append( np.std( R_20 ) )
    Sdefo20.append( np.sum( R_20 ) )
    DSdefo20.append( np.std( R_20 )*len(R_20)  )
    Mdefo20.append( np.max( R_20 )  )
    
    area.append( np.sum( mascara )*ps**2 )
    





#%%

data = pd.DataFrame()


data["Celula"] = cel

data["Defo0"] = defo00
data["P0"] = Pdefo00
data["DP0"] = DPdefo00
data["S0"] = Sdefo00
data["DS0"] = DSdefo00
data["M0"] = Mdefo00

data["Defo10"] = defo10
data["P10"] = Pdefo10
data["DP10"] = DPdefo10
data["S10"] = Sdefo10
data["DS10"] = DSdefo10
data["M10"] = Mdefo10

data["Defo20"] = defo20
data["P20"] = Pdefo20
data["DP20"] = DPdefo20
data["S20"] = Sdefo20
data["DS20"] = DSdefo20
data["M20"] = Mdefo20

data["Area"] = area
data["Clase"] = ["MCF7SS"]*len(cel )


data.to_csv( "data_MCF7SS.csv" )




#%% 
# pre_post7.update({(1,1):(8,7)})
# pre_post7.update({(1,1):(6,5)})

tup = (7,2)


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
mascara10 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_10um.png")
# mascara = np.loadtxt( path + r"PIV\Mascaras MCF7\\" + name + "_m_00um.png")

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
    
if tup == (1,1):
    delta = 3
    pre_profundo = stack_pre[ pre_post7[tup][0] + delta ]
    post_profundo = correct_driff( stack_post[ pre_post7[tup][1] + delta ], pre_profundo, 50 )
    
    sat = np.zeros([1024]*2)
    sat[ pre > 1000 ] = 1
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    
    pre = pre*(1-sat) + pre_profundo*sat
    post = post*(1-sat) + post_profundo*sat
    
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
plt.imshow( mascara, cmap = cm_B, alpha = 0.6 )

plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.show()


# mascara10 = np.loadtxt( path + r"PIV\Mascaras MCF7\\" + name + "_m_10um.png")
# if mascara10[1,1] == 1:
#     mascara10 = 1 - mascara10    
# mascara20 = np.loadtxt( path + r"PIV\Mascaras MCF7\\" + name + "_m_20um.png")
# if mascara20[1,1] == 1:
#     mascara20 = 1 - mascara20

# Y_work, X_work = Y_s, X_s
# Y_cell_00 = []
# X_cell_00 = []
# Y_cell_10 = []
# X_cell_10 = []
# Y_cell_20 = []
# X_cell_20 = []
    
# l = len(Y_work)
# ps = pixel_size
# for j in range(l):
#     for i in range(l):
#         if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara[ int(x[j,i]), int(y[j,i]) ]) == 1:
#             Y_cell_00.append(Y_work[j,i])
#             X_cell_00.append(X_work[j,i])
#         if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
#             Y_cell_10.append(Y_work[j,i])
#             X_cell_10.append(X_work[j,i]) 
#         if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara20[ int(x[j,i]), int(y[j,i]) ]) == 1:
#             Y_cell_20.append(Y_work[j,i])
#             X_cell_20.append(X_work[j,i])


















































#%%


tup = (11,4)


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
mascara10 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_10um.png")
# mascara = np.loadtxt( path + r"PIV\Mascaras MCF7\\" + name + "_m_00um.png")

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
    
if tup == (1,1):
    delta = 3
    pre_profundo = stack_pre[ pre_post7[tup][0] + delta ]
    post_profundo = correct_driff( stack_post[ pre_post7[tup][1] + delta ], pre_profundo, 50 )
    
    sat = np.zeros([1024]*2)
    sat[ pre > 1000 ] = 1
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    
    pre = pre*(1-sat) + pre_profundo*sat
    post = post*(1-sat) + post_profundo*sat
    
vi = int( int( 3/ps )*4 )
it = 3
# bordes_extra = 10 # px
bordes_extra = int(np.round(vi/9))

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
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

plt.subplot(1,2,1)
plt.imshow( celula_pre -fondo, cmap = 'gray' )
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")

plt.subplot(1,2,2)
plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = cm_O, alpha = 0.6 )
plt.imshow( 1-mascara10, cmap = cm_O, alpha = 0.6 )


plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.show()


# mascara10 = np.loadtxt( path + r"PIV\Mascaras MCF7\\" + name + "_m_10um.png")
# if mascara10[1,1] == 1:
#     mascara10 = 1 - mascara10    
# mascara20 = np.loadtxt( path + r"PIV\Mascaras MCF7\\" + name + "_m_20um.png")
# if mascara20[1,1] == 1:
#     mascara20 = 1 - mascara20

# Y_work, X_work = Y_s, X_s
# Y_cell_00 = []
# X_cell_00 = []
# Y_cell_10 = []
# X_cell_10 = []
# Y_cell_20 = []
# X_cell_20 = []
    
# l = len(Y_work)
# ps = pixel_size
# for j in range(l):
#     for i in range(l):
#         if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara[ int(x[j,i]), int(y[j,i]) ]) == 1:
#             Y_cell_00.append(Y_work[j,i])
#             X_cell_00.append(X_work[j,i])
#         if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
#             Y_cell_10.append(Y_work[j,i])
#             X_cell_10.append(X_work[j,i]) 
#         if  0 < int(x[j,i]) < resolution and 0 < int(y[j,i]) < 1024 and int(mascara20[ int(x[j,i]), int(y[j,i]) ]) == 1:
#             Y_cell_20.append(Y_work[j,i])
#             X_cell_20.append(X_work[j,i])


















