# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:38:50 2023

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

#%%
region = 1
metadata = pd.read_csv('Data.csv', delimiter = ',', usecols=np.arange(3,15,1))
otradata = pd.read_csv('Data.csv', delimiter = ',', usecols=np.arange(0,2,1), index_col=0, header=0, names = ['1','2'])
metadata_region = metadata.loc[ metadata["Región"] == region ]

stack_pre = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
celula = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'TRANS' ]["Archivo"].values[0]+".oif" )[1]
celula_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1][0]
mascara = 1 - iio.imread( "mascara_" + str(region) + ".png" )

field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

pre = stack_pre[ 5 ]
post, ZYX = correct_driff_3D( stack_post, pre, 50, info = True)

print(metadata_region["Archivo"].values)
# print(n_pre, n_post  )
print(ZYX)

a = np.mean(post)/np.mean(pre)

plt.figure()
plt.title('Pre')
plt.imshow( pre, cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post, cmap = 'gray', vmin = 80*a, vmax = 700*a  )

plt.figure()
plt.title('Trans')
plt.imshow( celula , cmap = 'gray' )

plt.figure()
plt.title('Trans')
plt.imshow( celula_post , cmap = 'gray' )

# %% PIV + NMT + Suavizado

vi = 200
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

#%%

plt.imshow(np.sqrt(X_s**2 + Y_s**2))
# plt.imshow( mascara, cmap = "Oranges", alpha = 0.3 )
plt.title(str(vi))    
#%%

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = "Oranges", alpha = 0.9 )
plt.quiver(x,y,X_s,-Y_s, scale = 100, pivot='tail')
plt.xlim([0, resolution])
plt.ylim([resolution,0])

#%%

plt.hist( R_s.flatten()*pixel_size, bins = np.arange(-0.15, 11.15, 0.3)*pixel_size )
plt.grid( True )

#%% Plot

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
start_x = d + 0  # Starting x-coordinate of the scale bar
start_y = resolution -( 2*wind )# Starting y-coordinate of the scale bar

plt.figure(figsize=(20,20), tight_layout=True)


# Transmision pre
plt.subplot(2,2,1)

plt.imshow( celula , cmap = 'gray' )
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")

# Transmision post
plt.subplot(2,2,2)

plt.imshow( celula_post , cmap = 'gray' )
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
plt.imshow( mascara, cmap = "Oranges", alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.show()

#%%

#signal.correlate(img_post - img_post.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft")/(img_shape**2)

#%%

n_pre = [6,5,4,8,5,4,4]
ccolors = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys", "Reds"]


defo = []
Ddefo = []

for region in range(1,8,1):
    metadata = pd.read_csv('Data.csv', delimiter = ',', usecols=np.arange(3,15,1))
    otradata = pd.read_csv('Data.csv', delimiter = ',', usecols=np.arange(0,2,1), index_col=0, header=0, names = ['1','2'])
    metadata_region = metadata.loc[ metadata["Región"] == region ]
    
    stack_pre = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
    stack_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
    celula = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'TRANS' ]["Archivo"].values[0]+".oif" )[1]
    celula_post = of.imread( metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1][0]
    mascara = 1 - iio.imread( "mascara_" + str(region) + ".png" )
    
    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution
    
    pre = stack_pre[ n_pre[region-1] ]
    post, ZYX = correct_driff_3D( stack_post, pre, 50, info = True)


    vi = 128
    it = 3
    bordes_extra = 10 # px
    
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 5
    # modo = "No control"
    
    modo = "Smooth3"
    # modo = "Fit"
    if region == 7:
        modo = "No control"
    
    
    mapas = False
    suave0 = 3
    
    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    R_s = np.sqrt( X_s**2 + Y_s**2 )
    x, y = dominio
    
    
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
    start_x = d + 0  # Starting x-coordinate of the scale bar
    start_y = resolution -( 2*wind )# Starting y-coordinate of the scale bar

    
    plt.figure()
    
    al = 0.5
    if region == 7:
        al = 0.3
    
    plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
    plt.imshow( mascara, cmap = ccolors[region-1], alpha = al )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')

    # plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,resolution])
    plt.ylim([resolution,0])
    
    
    print(field)
    # defo.append( np.mean( R_s ) )
    # Ddefo.append( np.std( R_s ) )
    defo.append( R_s*pixel_size )
#%%
plt.grid(True)
sns.violinplot(data=defo, x=None, y=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=500, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None)
plt.ylabel("Deformación [um]")
plt.xlabel("Célula")
plt.xticks([0,1,2,3,4,5,6],[1,2,3,4,5,6,7])



#%%

plt.errorbar( np.arange(1,8,1), defo, Ddefo, fmt = 'o'  )
plt.grid( True )

#%%

dr = 1 # um
ps = pixel_size
dr_px = dr/ps



th = 0.1
ks = int(np.round( dr_px/th/0.4 ))

#%%
m1 = np.zeros([1024]*2)
m1[:,500:700] = 1

m2 = area_upper(m1, kernel_size = ks, threshold = th)
msuma = m2+m1
plt.imshow(msuma)

#%%

plt.plot( np.diff(msuma[510]) )
msuma_diff = np.diff(msuma[510])
#%%
puntos_subida = []
puntos_bajada = []
for i in range( len(msuma_diff) ):
    if msuma_diff[i] > 0.5:
        puntos_subida.append( i )
    if msuma_diff[i] < -0.5:
        puntos_bajada.append( i )

print(np.diff(puntos_subida))
print(np.diff(puntos_bajada))

#%%
ker_list = []
incremento = []

for i in range(9, 51, 2):
    m1 = np.zeros([1024]*2)
    m1[:,500:700] = 1

    m2 = area_upper(m1, kernel_size = i, threshold = 0.1)
    msuma = m2+m1
    msuma_diff = np.diff(msuma[512])

    puntos_subida = []
    puntos_bajada = []
    for j in range( len(msuma_diff) ):
        if msuma_diff[j] > 0.5:
            puntos_subida.append( j )
        if msuma_diff[j] < -0.5:
            puntos_bajada.append( j )

    print(i)
    ker_list.append(i)
    incremento.append(np.diff(puntos_subida)[0])


#%%

plt.plot(ker_list, incremento, 'o')

m = np.mean(np.diff(np.array(incremento).flatten())/2)
print(m)
#%%

def recta(x, m, b):
    return m*x + b

popt, pcov = curve_fit(recta, ker_list, np.array(incremento).flatten())
print(popt, np.sqrt(np.diag(pcov)))


#%%















