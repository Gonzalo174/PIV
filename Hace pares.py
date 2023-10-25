# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:23:24 2023

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

#%%
plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 16
path = r"C:\Users\gonza\1\Tesis\2023\\"
# path = r"D:\Gonzalo\\"
#%% Import 10
carpetas = ["23.10.05 - gon MCF10 1 - A04", "23.10.05 - gon MCF10 2 - D04", "23.10.05 - gon MCF10 3 - E04", "23.10.06 - gon MCF10 4 - C04", "23.10.19 - gon MCF10 6 - G18", "23.10.20 - gon MCF10 7 - I18" ]
distribucion = [ 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6 ]
pre10 =  [ 8, 4, 5, 7, 4, 4, 4, 4, 4, 4, 5, 4, 5, 2, 4, 4, 4, 4, 4,  4, 6, 5, 6, 5, 5,  3, 4, 5, 5, 3 ]
post10 = [ 8, 4, 6, 6, 2, 5, 3, 3, 4, 2, 5, 4, 4, 4, 4, 5, 5, 4, 4,  4, 5, 7, 8, 4, 6,  4, 5, 6, 4, 4 ]
#%% Import 7
carpetas = ["23.08.17 - gon MCF7 1 - C16", "23.08.18 - gon MCF7 2 - B16", "23.08.18 - gon MCF7 3 - A16", "23.08.24 - gon MCF7 4 - A23", "23.08.24 - gon MCF7 5 - B23", "23.08.25 - gon MCF7 6 - D23", "23.08.25 - gon MCF7 7 - C23", "23.08.31 - gon MCF7 8 - B30", "23.08.31 - gon MCF7 9 - A30", "23.09.01 - gon MCF7 10 - C30", "23.09.01 - gon MCF7 11 - D30"]
muestras = [ "C16", "B16", "A16", "A23", "B23", "D23", "C23", "B30", "A30", "C30", "D30" ]
cs =  [ (11,2), (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
pre_post7 = {(11,2):(6,5), (11,3):(4,3), (11,4):(4,4), (10,1):(4,3), (10,2):(8,4), (10,5):(4,4), (9,1):(3,3), (1,1):(9,11),    (8,2):(4,4), (8,3):(4,4), (7,1):(6,4), (7,2):(5,4), (6,2):(6,5), (6,3):(5,4), (6,4):(4,5), (5,4):(3,4), (4,1):(7,7), (3,3):(6,6)   }

#%% 10
r = 29
full_path1 = path + carpetas[distribucion[r]-1]

name = carpetas[distribucion[r]-1][-3:] + "_R" + str(int(r))

metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == r ]

field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[0]
celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]

#%% 7
tup = (4,1)
full_path1 = path + carpetas[ distribucion[ tup[0] -1 ] ]

name = carpetas[ tup[0] - 1 ][-3:] + "_R0" + str( tup[1] )

metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
metadata_region = metadata.loc[ metadata["Región"] == tup[1] ]

field = metadata_region["Campo"].values[0]
resolution = metadata_region["Tamano imagen"].values[0]
pixel_size = field/resolution

stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]


#%%
err_pre = []
err_post = []

for i in range(len(stack_post)):
    err_post.append( np.std( stack_post[i] ) )
    
for i in range(len(stack_pre)):
    err_pre.append( np.std( stack_pre[i] )   )
    
plt.plot( normalizar(err_pre), label = "PRE" )
plt.plot( normalizar(err_post), label = "POST" )
plt.grid(True)
plt.legend()
plt.title(name)























