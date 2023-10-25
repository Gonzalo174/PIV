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

#%%
c0 = [(0, 0, 0), (1, 1, 1)]
cm0 = ListedColormap(c0)
path = r"D:\Gonzalo\\"
carpetas = ["23.10.05 - gon MCF10 1 - A04", "23.10.05 - gon MCF10 2 - D04", "23.10.05 - gon MCF10 3 - E04", "23.10.06 - gon MCF10 4 - C04", "23.10.19 - gon MCF10 6 - G18", "23.10.20 - gon MCF10 7 - I18" ]
distribucion = [ 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6 ]
# distribucion = [ 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3 ]

#%%

for r in range(20,31,1):
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
    celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,2+pre10[r-1]-post10[r-1]]
    
    plt.imsave( str( carpetas[distribucion[r]-1][-3:] ) + "_R" + str(r) + "_m.png", celula_pre, cmap = "gray" )
    # mascara = 1- iio.imread(path + r"PIV\Mascaras\\" + str( muestras[tup[0]-1] ) + "R" + str(tup[1]) + ".png")
    # mascara = mascara3



#%%

data = os.listdir()
print(data)


#%%

for i in range(20,31,1):
    name = data[i-20]
    print(i, int(name[5:7]))
    mascara = iio.imread(name)
    
    full_path1 = path + carpetas[ distribucion[i]-1 ]
    
    metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
    metadata_region = metadata.loc[ metadata["Región"] == int(name[5:7]) ]

    field = metadata_region["Campo"].values[0]
    resolution = metadata_region["Tamano imagen"].values[0]
    pixel_size = field/resolution
    
    
    dr = 21 # um
    ps = pixel_size
    ks = int(np.round( dr/ps/0.4 ))
    
    m0 = 1 - np.copy(mascara)
    np.savetxt(name[:-4] + '_00um.csv', m0)
    ms = np.copy(mascara)
    for j in range(dr):
        m0 = area_upper(m0, kernel_size = ks//dr, threshold = 0.1)
        ms = ms + m0
        # print(j)
        # plt.imshow(ms)
        if j == 10:
            # plt.imsave( name[:-4] + '_10um.png', 1-m0, cmap = cm0 )
            np.savetxt(name[:-4] + '_10um.csv', m0)
            
    # plt.imsave( name[:-4] + '_20um.png', 1-m0, cmap = cm0 )
    np.savetxt(name[:-4] + '_20um.csv', m0)
    
#%%
    
    
mascara = iio.imread(data[5])    
plt.imshow(mascara)
    
np.savetxt("prueba.csv", mascara)

#%%
m00 = np.loadtxt("A04_R01_m_00um.csv")
m10 = np.loadtxt("A04_R01_m_10um.csv")    
m20 = np.loadtxt("A04_R01_m_20um.csv")    
    
plt.imshow(m20+m10+m00)
plt.imsave( "p1.png", m20+m10+m00 )
    
#%%
m00 = np.loadtxt("E04_R11_m_00um.csv")
m10 = np.loadtxt("E04_R11_m_10um.csv")    
m20 = np.loadtxt("E04_R11_m_20um.csv")    
    
plt.imshow(m20+m10+m00)

plt.imsave( "p11.png", m20+m10+m00 )
    
#%%

for i in data[6:]:
    img = iio.imread(i)
    if len( img.shape ) > 2:
        img = img[:,:,0]
    np.savetxt(i, 1 - img/np.max(img) )
    
#%%
m0 = iio.imread("A16R3_10um_SS.png")
#%%
np.savetxt("A16_R03_m_10um.png", m0[:,:,0]/255)
    
    
    
    
    
    
