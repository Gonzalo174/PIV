# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:18:22 2023

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

#MCF7 D30 R4 cel9 del 1/9   0
#MCF7 C30 R5 cel5 del 1/9   1
#MCF10 D04 R9 5/10          2
#MCF10 G18 R25 del 19/10    3
#%% Invocacion

cel = 0
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

plt.figure( figsize=[12,4] )
desvios = desvio_por_plano(stack_pre)
plt.plot(desvios, 'o', )
plt.grid()
plt.xticks(np.arange(len(stack_pre)))
plt.show()
#%%
pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

VM = 600
Vm = 100
cm = cm_crimson
fs = 'xx-large'
alpha = np.mean( pre )/np.mean( post )

plt.figure(figsize = [14, 14], tight_layout=True)

plt.subplot(2,2,1)
plt.imshow( pre, cmap = cm, vmin = Vm, vmax = VM )
barra_de_escala( 10, sep = 1.5,  font_size = fs )

plt.subplot(2,2,2)
plt.imshow( post, cmap = cm, vmin = Vm, vmax = VM/alpha )
barra_de_escala( 10, sep = 1.5,  font_size = fs )

plt.subplot(2,2,3)
plt.imshow( pre, cmap = cm, vmin = Vm + 50, vmax = VM - 200 )
plt.imshow( post, cmap = cm_green, vmin = Vm + 50, vmax = VM/alpha - 200, alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', lw = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = fs )

plt.subplot(2,2,4)
plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1], b[0], c = 'w', lw = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )

plt.show()

print(data)

#%%
A = As[cel]
desvios_bin, limit = busca_esferas( pre, ps = ps, th = A )
plt.figure( figsize = [14, 14] )
plt.imshow( pre[ :limit , :limit ], cmap = cm, vmin = 150, vmax = 600 )
plt.imshow( desvios_bin, cmap = 'gray', alpha = 0.09, extent = [0,limit,limit,0])
barra_de_escala( 10, sep = 2, more_text = 'Célula ' + str(cel), a_lot_of_text = str( int( np.mean( desvios_bin )*100 ) ) + '%', font_size = 'xx-large' )

#%%
it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = int(np.round(vi/2**(it-1))/4) 

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
# modo = "Fit"
mapas = False
suave0 = 3

j_p, i_p = np.random.randint( int( len( pre )/(3/ps) )  - 2 ) + 1, np.random.randint( int( len( pre )/(3/ps) )  - 2 ) + 1
control0 = [(j_p,i_p)]
# control0 = [(0,0)]
# control0 = [(-1,-1)]

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = A, control = control0)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

#%%
scale0 = 100
plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X_nmt**2 + Y_nmt**2) )*ps, 2)  )


#%%
scale0 = 100
plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round(np.max( np.sqrt(X_s**2 + Y_s**2) )*ps, 2)  )


#%%

R_nmt = np.sqrt(X_nmt**2 + Y_nmt**2)*ps
plt.imshow(R_nmt)
plt.colorbar()


#%%

R_s = np.sqrt(X_s**2 + Y_s**2)*ps
plt.imshow(R_s)
plt.colorbar()
#%%
# scale0 = 100
# plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
# plt.quiver(x,y,deformacion[1], -deformacion[0], scale = scale0, pivot='tail')
# barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
# plt.xlim([0,1023])
# plt.ylim([1023,0])

# print( np.round(np.max( np.sqrt(deformacion[1]**2 + deformacion[0]**2) )*ps, 2)  )



#%% Dependencia Axial de la deformación
z0 = 2

l = int( int( 1024//vi + 1 )*4 )
Yz = np.zeros([len(stack_pre)-z0, l, l])
Xz = np.zeros([len(stack_pre)-z0, l, l])

Yz_s = np.zeros([len(stack_pre)-z0, l, l ])
Xz_s = np.zeros([len(stack_pre)-z0, l, l ])

it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = int(np.round(vi/2**(it-1)/3)) 

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
suave0 = 3

mascara_grosa = np.zeros( [l]*2 )
for j in range(l):
    for i in range(l):
        if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
            mascara_grosa[i,j] = 1

for z in range(z0,len(stack_pre)-1,1):
    pre = stack_pre[z]
    post = correct_driff( stack_post[z], pre, 50 )
    # post, data = correct_driff_3D( stack_post, pre, 50, info = True )

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = A)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio

    Xz_s[z-z0] = X_s*mascara_grosa
    Yz_s[z-z0] = Y_s*mascara_grosa

    Xz[z-z0] = X_nmt*mascara_grosa
    Yz[z-z0] = Y_nmt*mascara_grosa
    
    scale0 = 100
    # plt.figure()
    
    
    plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
    # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'Z = ' + str(z) )
    plt.xlim([0,1023])
    plt.ylim([1023,0])
    
#%%

Rz = np.sqrt(Xz_s**2 + Yz_s**2)
Rz_dist = np.array( [ Rz[z].flatten()*ps for z in range( len(Rz) ) ] )
df = pd.DataFrame()
for i in range(len(Rz_dist)-1):
    df[ str(i/2 - 1) ] = Rz_dist[i]

#%%
plt.figure(figsize=[12,4])
sns.violinplot( df  )
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Profundidad [µm]" )
plt.grid()

#%%
# plt.hist( Rz_dist[2], bins=np.arange(-0.025, 1.025, 0.05) )

Rz_mean = np.mean( np.mean( Rz, axis = 1 ), axis = 1 )*ps
Rz_std = np.array( [ np.std( Rz[z] ) for z in range( len(Rz) -1) ] )*ps
Rz_max = np.array( [ np.max( Rz[z] ) for z in range( len(Rz) -1) ] )*ps

plt.plot(Rz_mean[:-1])
# plt.plot(Rz_std)
# plt.plot(Rz_max)

#%%














#%% Dependencia con la ventana de exploración de al deformación
z0 = 5

# bordes_extra = int(np.round(vi/2**(it-1)/4)) 
bordes_extra = 10
# ventanas = np.arange(4.7,2.6,-0.1) 
ventanas = np.arange(60,24,-1)  # en px

# Rz = np.zeros([len(ventanas)])
Rz_s = []              
ceros_lista = []

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
suave0 = 3

pre = stack_pre[z0]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

df = pd.DataFrame()

for vs in ventanas:
    it = 3
    vi = int( int( vs )*2**(it-1) )

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = A)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio
    
    l = len(X_s)
    mascara_grosa = np.zeros( [l]*2 )
    for j in range(l):
        for i in range(l):
            if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                mascara_grosa[i,j] = 1
    
    
    Rz = np.sqrt((X_nmt*mascara_grosa)**2 + (Y_nmt*mascara_grosa)**2)*ps
    Rz_plano = Rz.flatten() 
    ceros = 0
    for r in Rz_plano:
        if r == 0:
            ceros += 1
    
    # Rz = np.sqrt((X_nmt)**2 + (Y_nmt)**2)*ps

    Rz_s.append( Rz_plano ) 
    ceros_lista.append( (len(Rz_plano) - ceros) )
    
    # scale0 = 100
    # plt.figure()
    # plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
    # # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
    # plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    # barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'V = ' + str(np.round(vs,1)) )
    # plt.xlim([0,1023])
    # plt.ylim([1023,0])
    # plt.show()
    
#%%
plt.figure(figsize=[12,4])
sns.violinplot( Rz_s )
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Ventana final [µm]" )
plt.xticks( np.arange(len(ventanas)), np.round( ventanas, 1 ) )
plt.grid()

#%%
plt.figure( figsize = [12,4] )
plt.plot( ventanas, [np.mean(Rz_s[i])*len(Rz_s[i])/( len(Rz_s[i]) - ceros_lista[i] ) for i in range(len(Rz_s))] , 'o' )
plt.ylabel( "Deformación promedio [µm]" )
plt.xlabel( "Ventana final [px]")#[µm]" )
# plt.xticks( np.arange(len(ventanas)), np.round( ventanas, 1 ) )
plt.xlim([61,24])
plt.grid()



#%%
plt.quiver(x,y,X_nmt,Y_nmt, scale = scale0, pivot='tail')
plt.quiver(x,y,X_nmt*mascara_grosa,Y_nmt*mascara_grosa, scale = scale0, pivot='tail')




#%%
np.mean(  np.abs(Y_nmt)  )
np.mean(  np.abs(Y_nmt*mascara_grosa)  )

#%% estaba teniendo en cuenta los ceros igual...
plt.hist( np.abs(Y_nmt).flatten() )






#%% Deformación Axial

it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = int(np.round(vi/2**(it-1))/4) 

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
# modo = "Fit"
suave0 = 3

j_p, i_p = np.random.randint( len(x) - 2 ) + 1, np.random.randint( len(x) - 2 ) + 1
control0 = [(j_p,i_p)]
# control0 = [(-1,-1)]

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = A, control = control0 )
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

#%%
scale0 = 100
plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X_nmt**2 + Y_nmt**2) )*ps, 2)  )

#%%
scale0 = 100
plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X_s**2 + Y_s**2) )*ps, 2)  )

#%%
z_0 = 0
Z = Z_iteration0( stack_post0 = stack_post, img_pre0 = pre, win_shape = int( 3/ps ), exploration = 1, translation_Y = Y_nmt, translation_X = X_nmt, mode = "Smooth3", A = As[cel], z0 = z_0 )
j_p, i_p = np.random.randint( len(x) - 2 ) + 1, np.random.randint( len(x) - 2 ) + 1
# j_p, i_p = -1, -1

#%%
cm_semaforo = ListedColormap( [(200/255, 10/255, 30/255), (1, 0.9, 0), (0, 128/255, 0)] ) 
plt.figure()
plt.imshow(Z, cm_semaforo)
plt.colorbar()

#%%

scale0 = 100
plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
plt.quiver(x,y,X + 1,-Y-1, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X**2 + Y**2) )*ps, 2)  )


#%%



#%%





































