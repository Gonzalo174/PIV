"""
Created on Mon Jun 26 09:35:01 2023

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


#%% Import

CS =      [ (11,2), (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
pre_CS =  [      6,      4,      4,       4,      8,      4,     3,     9 ]
post_CS = [      5,      3,      4,       3,      4,      4,     3,    11 ]

SS =      [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
pre_SS =  [     4,     4,     6,     5,     6,     5,     4,     3,     7,     6 ]
post_SS = [     4  ,   4,     4,     4,     5,     4,     5,     4,     8,     6 ]

Cualitativas = [ (8,1), (8,5), (7,2), (6,1), (5,3), (5,5), (3,5) ]

path = r"C:\Users\gonza\1\Tesis\2023\\"
carpetas = ["23.08.17 - gon MCF7 1", "23.08.18 - gon MCF7 2", "23.08.18 - gon MCF7 3", "23.08.24 - gon MCF7 4", "23.08.24 - gon MCF7 5", "23.08.25 - gon MCF7 6", "23.08.25 - gon MCF7 7", "23.08.31 - gon MCF7 8 - B30", "23.08.31 - gon MCF7 9 - A30", "23.09.01 - gon MCF7 10 - C30", "23.09.01 - gon MCF7 11 - D30"]
muestras = [ "C16", "B16", "A16", "A23", "B23", "D23", "C23", "B30", "A30", "C30", "D30" ]


#%% Guardar las imagenes de las celulas para hacer las mascaras

for tup in SS:
    full_path = path + carpetas[int(tup[0]-1)] + '\Data.csv'
    region = int( tup[1]  )
    metadata = pd.read_csv( full_path, delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == region ]

    celula = of.imread( path + carpetas[int(tup[0])-1] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'TRANS' ]["Archivo"].values[0]+".oif" )[1]


    plt.figure()
    plt.imshow( celula, cmap = "gray" )
    
    name =  muestras[tup[0]-1] + 'R' + str(tup[1])
    plt.title(name)
    
    print(name)
    plt.imsave( name + '_celula_SS.png', celula )


#%% Generar y guardar las mascaras incrementadas

for tup in SS[-2:-1]:
    full_path = path + carpetas[int(tup[0]-1)] + '\Data.csv'
    region = int( tup[1]  )
    metadata = pd.read_csv( full_path, delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == region ]

    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution

    name =  muestras[tup[0]-1] + 'R' + str(tup[1])
    print(name)
    
    mascara0 = 1- iio.imread(r"C:\Users\gonza\1\Tesis\PIV\Mascaras\\" + name + "_SS.png")
    mascara = np.zeros([1024]*2)
    mascara[:1020] = mascara0
    
    dr = 21 # um
    ps = pixel_size
    ks = int(np.round( dr/ps/0.4 ))
    
    m0 = np.copy(mascara)
    ms = np.copy(mascara)
    for j in range(dr):
        m0 = area_upper(m0, kernel_size = ks//dr, threshold = 0.1)
        ms = ms + m0
        print(j)
        plt.imshow(ms)
        if j == 10:
            plt.imsave( name + '_10um_SS.png', 1-m0, cmap = 'gray' )
            
    plt.imsave( name + '_20um_SS.png', 1-m0, cmap = 'gray'  )


#%% Ver los perfiles en Z y elegir el plano mas superficial

for tup in CS:
    full_path = path + carpetas[int(tup[0]-1)] + '\Data.csv'
    region = int( tup[1]  )
    metadata = pd.read_csv( full_path, delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == region ]

    stack_pre = of.imread( path + carpetas[int(tup[0]-1)] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( path + carpetas[int(tup[0])-1] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]

    pla_pre = []
    err_pre = []
    pla_post = []
    err_post = []

    for i in range(len(stack_post)):
        err_post.append( np.std( stack_post[i] ) )
        pla_post.append( i )

    for i in range(len(stack_pre)):
        err_pre.append( np.std( stack_pre[i] )   )
        pla_pre.append( i )
    
    name =  muestras[tup[0]-1] + 'R' + str(tup[1])
    
    plt.figure()
    plt.title(name + str( tup ) )        
    plt.plot( pla_pre, (err_pre-min(err_pre))/max((err_pre-min(err_pre))), label = "PRE")
    plt.plot( pla_post, (err_post-min(err_post))/max((err_post-min(err_post))), label = "POST")
    plt.legend()
    plt.grid(True)    

    print(name)

#%% Control de los planos sobre los que se realiza PIV

for i, tup in enumerate(CS):
    full_path = path + carpetas[int(tup[0]-1)] + '\Data.csv'
    region = int( tup[1]  )
    metadata = pd.read_csv( full_path, delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == region ]

    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution

    stack_pre = of.imread( path + carpetas[int(tup[0]-1)] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( path + carpetas[int(tup[0])-1] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
    mascara = 1- iio.imread(r"C:\Users\gonza\1\Tesis\PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + ".png")

    pre = stack_pre[ pre_CS[i] ]
    post0 = stack_post[ post_CS[i] ]
    post, m, YX = correct_driff( post0 , pre, 50, info = True)
    
    vi = 100
    it = 3
    bordes_extra = 10 # px

    Noise_for_NMT = 0.2
    Threshold_for_NMT = 5
    # modo = "No control"
    modo = "Smooth3"

    mapas = False
    suave0 = 3

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    R_s = np.sqrt( X_s**2 + Y_s**2 )
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
    start_x = d + 80  # Starting x-coordinate of the scale bar
    start_y = resolution -( 2*wind )# Starting y-coordinate of the scale bar

    plt.figure(figsize=(20,20), tight_layout=True)

    # Pre
    plt.subplot(2,2,1)

    plt.imshow( pre, cmap = 'gray', vmin = 80, vmax = 700)
    plt.xticks([])
    plt.yticks([])
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
    plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")

    # Post
    plt.subplot(2,2,2)

    plt.imshow( post, cmap = 'gray', vmin = 80*a, vmax = 700*a)
    plt.xticks([])
    plt.yticks([])
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

    # Esferas
    plt.subplot(2,2,3)
    plt.imshow( np.zeros(pre.shape), cmap = cm0 )
    plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
    plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)
    plt.xticks([])
    plt.yticks([])
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)

    # Deformacion
    plt.subplot(2,2,4)

    plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
    plt.imshow( mascara, cmap = cm3, alpha = 0.5 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

    # plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,resolution])
    plt.ylim([resolution,0])

    plt.show()

 
    print(name)


#%% Guardar las distribuciones de defromacion

defo0  = []
defo10 = []
defo20 = []

for i, tup in enumerate(CS):
    full_path = path + carpetas[int(tup[0]-1)] + '\Data.csv'
    region = int( tup[1]  )
    metadata = pd.read_csv( full_path, delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == region ]

    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution

    stack_pre = of.imread( path + carpetas[int(tup[0]-1)] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( path + carpetas[int(tup[0])-1] + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
    mascara   = 1- iio.imread(r"C:\Users\gonza\1\Tesis\PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + ".png")
    mascara10 = 1- iio.imread(r"C:\Users\gonza\1\Tesis\PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + "_10um.png")[:,:,0]/255
    mascara20 = 1- iio.imread(r"C:\Users\gonza\1\Tesis\PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + "_20um.png")[:,:,0]/255
    nom = str( muestras[tup[0]-1] ) + "R" + str(tup[1])
    print(nom)
    pre = stack_pre[ pre_CS[i] ]
    post0 = stack_post[ post_CS[i] ]
    post, m, YX = correct_driff( post0 , pre, 50, info = True)
    
    vi = 100
    it = 3
    bordes_extra = 10 # px

    Noise_for_NMT = 0.2
    Threshold_for_NMT = 5
    modo = "Smooth3"
    suave0 = 3

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    R_s = np.sqrt( X_s**2 + Y_s**2 )
    x, y = dominio

    plt.figure()
    plt.title(nom)
    plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
    plt.imshow( mascara, cmap = cm3, alpha = 0.5 )
    plt.imshow( mascara10, cmap = cm3, alpha = 0.5 )
    plt.imshow( mascara20, cmap = cm3, alpha = 0.5 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,resolution])
    plt.ylim([resolution,0])
    
    
    Y_work, X_work = np.copy( Y_s ), np.copy( X_s )
    l = len(Y_work)
    
    Y_cell  , X_cell   = [], []
    Y_cell10, X_cell10 = [], []
    Y_cell20, X_cell20 = [], []

    for j in range(l):
        for i in range(l):
            if 0 < x[j,i] < 1023 and 0 < y[j,i] < 1023 and mascara[ int(x[j,i]), int(y[j,i]) ] == 1:
                # print(x[j,i])
                Y_cell.append(Y_work[j,i])
                X_cell.append(X_work[j,i])
            if 0 < x[j,i] < 1023 and 0 < y[j,i] < 1023 and mascara10[ int(x[j,i]), int(y[j,i]) ] == 1:
                Y_cell10.append(Y_work[j,i])
                X_cell10.append(X_work[j,i])
            if 0 < x[j,i] < 1023 and 0 < y[j,i] < 1023 and mascara20[ int(x[j,i]), int(y[j,i]) ] == 1:
                Y_cell20.append(Y_work[j,i])
                X_cell20.append(X_work[j,i])
            
    defo0.append(  np.sqrt( np.array(Y_cell)**2 + np.array(X_cell)**2 )*pixel_size )
    defo10.append(  np.sqrt( np.array(Y_cell10)**2 + np.array(X_cell10)**2 )*pixel_size )
    defo20.append(  np.sqrt( np.array(Y_cell20)**2 + np.array(X_cell20)**2 )*pixel_size )

#%%

plt.grid(True)
plt.title('20 um')
sns.violinplot(data=defo20, x=None, y=None, bw='scott', cut=0, scale='area', scale_hue=True, gridsize=500, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None)
plt.ylabel("Deformación [um]")
plt.xlabel("Célula")
plt.xticks([0,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8])





#%%

c0 = [(0, 0, 0), (0, 0, 0)]
cm0 = ListedColormap(c0)
c3 = [(1, 1, 1), (1, 1, 0)]
cm3 = ListedColormap(c3)

c1 = []
c2 = []
for i in range(1000):
    c1.append( (i/999,0,0) )
    c2.append( (0,i/999,0) )

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)
