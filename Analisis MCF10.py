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
# plt.rcParams['font.family'] = "Yu Gothic"
cm_crimson = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255)) for i in range(1000)] )
cm_green = ListedColormap( [(0,128*i/(999*255),0) for i in range(1000)] )
cm_yellow = ListedColormap( [( (220*i/(999*255)),128*i/(999*255),0) for i in range(1000)] )
cm_y = ListedColormap( [(1, 1, 1), (1, 1, 0)] )   # Blanco - Amarillo
cm_ar = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (0.839, 0.152, 0.157)] ) 
cm_aa = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (1.000, 0.498, 0.055)] ) 
cm_aa2 = ListedColormap( [(0.122, 0.467, 0.706), (0, 0, 0), (1.000, 0.498, 0.055)] ) 

cm0 = ListedColormap( [(0, 0, 0), (0, 0, 0)] )               # Negro
cm1 = ListedColormap( [(i/999,0,0) for i in range(1000)] )   # Negro - Rojo
cm2 = ListedColormap( [(0,i/999,0) for i in range(1000)] )   # Negro - Verde
cm3 = ListedColormap( [(1, 1, 1), (1, 1, 0)] )               # Blanco - Amarillo

#%% Import
path = r"C:\Users\gonza\1\Tesis\2023\\"
# path = r"D:\Gonzalo\\"
carpetas = ["23.10.05 - gon MCF10 1 - A04", "23.10.05 - gon MCF10 2 - D04", "23.10.05 - gon MCF10 3 - E04", "23.10.06 - gon MCF10 4 - C04", "23.10.19 - gon MCF10 6 - G18", "23.10.20 - gon MCF10 7 - I18" ]
distribucion = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5 ]
pre10 =  [ 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 2, 4, 4, 4, 4, 4,  4, 6, 5, 6, 5, 5,  3, 4, 5, 5, 3 ]
post10 = [ 4, 4, 6, 3, 2, 5, 3, 3, 4, 2, 5, 4, 4, 4, 4, 5, 5, 4, 4,  4, 5, 7, 8, 4, 6,  4, 5, 6, 4, 4 ]


cs = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 22, 25 ] 
ss = [14, 15, 17, 18, 26, 27, 28, 29]

#%%
ws = 3

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


for r in cs:
    full_path1 = path + carpetas[ distribucion[r-1] ]

    name = carpetas[ distribucion[r-1] ][-3:] + "_R" + str(int( 100 + r ))[1:]
    print(name)
    metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == r ]

    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    # pixel_size = field/resolutio
    zoom = metadata_region["Zoom"].values[0]
    pixel_size = 1/(4.97*zoom)

    stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
    celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
    celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,2+pre10[r-1]-post10[r-1]]
    mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_00um.csv")
    
    pre = stack_pre[ pre10[r-1] ]
    post = correct_driff( stack_post[ post10[r-1] ], pre, 50 )
    

    # vi = 128
    vi = int( int( ws/ps )*4 )
    it = 3
    # bordes_extra = 10 # px
    bordes_extra = int(np.round(vi/12))
    if r == 6 or r == 22 or r == 25:
        bordes_extra = int(1/ps)
        
    if r == 1 or r == 4 or r == 8 or r == 14 or r == 15 or r == 17 or r == 18:
        delta = 4
        if r == 1 or r == 17:
            delta = 6
        pre_profundo = stack_pre[ pre10[r-1] + delta ]
        post_profundo = correct_driff( stack_post[ post10[r-1] + delta ], pre_profundo, 50 )
        sat = busca_manchas(pre, 700)
        pre = pre*(1-sat) + pre_profundo*sat
        post = post*(1-sat) + post_profundo*sat    
    

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
    post_plot[ post < 0 ] = 0

    scale0 = 100
    scale_length = 10  # Length of the scale bar in pixels
    scale_pixels = scale_length/pixel_size
    scale_unit = 'µm'  # Unit of the scale bar

    wind = vi/( 2**(it-1) )
    d = int( ( resolution - len(Y_nmt)*wind )/2   )


    plt.figure(figsize=(7,7), layout='compressed')

    plt.subplot(2,2,1)
    plt.imshow( celula_pre , cmap = 'gray' )
    barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'w', more_text = str(name) )

    plt.subplot(2,2,2)
    plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
    plt.imshow( mascara, cmap = cm3, alpha = 0.6 )
    plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
    barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'k', text = False )


    plt.subplot(2,2,3)
    plt.imshow( np.zeros(pre.shape), cmap = cm0 )
    plt.imshow( pre_plot, cmap = cm_crimson, vmin = 0, vmax = 250, alpha = 1)
    plt.imshow( post_plot, cmap = cm_green, vmin = 0, vmax = 250, alpha = 0.5)
    barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'w', text = False )


    plt.subplot(2,2,4)
    plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
    plt.imshow( mascara, cmap = cm3, alpha = 0.6 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'k', text = False )


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
data["Clase"] = ["MCF10CS"]*len(cel )


data.to_csv( "data_MCF10CS.csv" )






















#%%
def busca_manchas(img, th = 800):
    sat = np.zeros([1024]*2)
    sat[ pre > th ] = 1
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    return sat

manchas = busca_manchas(pre)
plt.imshow(manchas)

#%% PIV + NMT + Suavizado


# pre10[0] = 8
# post10[0] = 8
# delta = 5

r = 17
full_path1 = path + carpetas[ distribucion[r-1] ]

name = carpetas[ distribucion[r-1] ][-3:] + "_R" + str(int( 100 + r ))[1:]
print( name ) 

metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == r ]

field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
# pixel_size = field/resolutio
zoom = metadata_region["Zoom"].values[0]
pixel_size = 1/(4.97*zoom)
ps = pixel_size


stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,2+pre10[r-1]-post10[r-1]]
mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_00um.csv")
# mascara = np.loadtxt( path + r"PIV\Mascaras MCF10\\" + name + "_m_00um.csv")

pre = stack_pre[ pre10[r-1] ]
post = correct_driff( stack_post[ post10[r-1] ], pre, 50 )
   
if r == 1:
    delta = 5
    pre_profundo = stack_pre[ pre10[r-1] + delta ]
    post_profundo = correct_driff( stack_post[ post10[r-1] + delta ], pre_profundo, 50 )
    
    sat = np.zeros([1024]*2)
    sat[ pre > 1000 ] = 1
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    # plt.imshow(sat)
    pre = pre*(1-sat) + pre_profundo*sat
    post = post*(1-sat) + post_profundo*sat
    
if r == 4:
    pre_profundo = stack_pre[ pre10[r-1] + delta ]
    post_profundo = correct_driff( stack_post[ post10[r-1] + delta ], pre_profundo, 50 )
    
    sat = np.zeros([1024]*2)
    sat[ pre > 1000 ] = 1
    for u in range(4):
        sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    # plt.imshow(sat)
    pre = pre*(1-sat) + pre_profundo*sat
    post = post*(1-sat) + post_profundo*sat

if r == 17:
    delta = 6
    sat = busca_manchas(pre)
    pre_profundo = stack_pre[ pre10[r-1] + delta ]
    post_profundo = correct_driff( stack_post[ post10[r-1] + delta ], pre_profundo, 50 )
    pre = pre*(1-sat) + pre_profundo*sat
    post = post*(1-sat) + post_profundo*sat




# vi = 128
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
post_plot[ post < 0 ] = 0

plt.figure(figsize=(7,7), layout='compressed')

plt.subplot(2,2,1)
plt.imshow( celula_pre , cmap = 'gray' )
barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'w', more_text = str(name) )

plt.subplot(2,2,2)
plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = cm3, alpha = 0.6 )
plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'k', text = False )


plt.subplot(2,2,3)
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
plt.imshow( pre_plot, cmap = cm_crimson, vmin = 0, vmax = 250, alpha = 1)
plt.imshow( post_plot, cmap = cm_green, vmin = 0, vmax = 250, alpha = 0.5)
barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'w', text = False )


plt.subplot(2,2,4)
plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = cm3, alpha = 0.6 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 20, pixel_size = ps,  sep = 1.5,  font_size = fs, color = 'k', text = False )


plt.show()

# plt.savefig(name + '_figura.png')


# c_pre_a = np.reshape( celula, (1, 1, *pre.shape)  )
# c_post_a = np.reshape( celula_post, (1, 1, *pre.shape)  )

# pre_a = np.reshape( pre, (1, 1, *pre.shape)  )
# post_a = np.reshape( post, (1, 1, *pre.shape)  ).astype(dtype=np.uint16)


# archivo1 = np.concatenate( (c_pre_a,c_post_a), axis = 1 )
# archivo2 = np.concatenate( (pre_a,post_a) , axis = 1 )

# archivo0 = np.concatenate( (archivo1,archivo2), axis = 0 )


print ( np.max( np.sqrt( np.array(Y_s)**2 + np.array(X_s)**2 )*ps ) )

#%% G = 8.64 por construcción, E = (31.6+1.8) medido, E = 2G(1+v)

fX, fY = ffttc_traction(X_s, Y_s, ps, ps, 31.6e3, sigma=0.83, filter="gaussian")#, fs = mascara )


#%%

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = "Reds", alpha = 0.4 )
# plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

plt.quiver(x,y,fX,-fY, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

# plt.savefig(name + '_figura.png')
plt.show()






