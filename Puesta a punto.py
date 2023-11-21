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

c1 = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
c2 = (1.0, 0.4980392156862745, 0.054901960784313725)
c3 = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
c4 = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)

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











#%%
cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')
print(ps)

#%% Dependencia Axial de la deformación
z0 = 2

it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = 8#int(np.round(vi/2**(it-1)/3)) 

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
suave0 = 3
fs = 'x-large'

# l = int( int( 1024//vi + 1 )*4 )
zf = min(len(stack_pre),len(stack_post))-1

pre = stack_pre[5]
post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )
delta_z = ZYX[0] - 5

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
x, y = dominio

l = len(x)

Yz = np.zeros([zf-z0, l, l])
Xz = np.zeros([zf-z0, l, l])

Yz_s = np.zeros([zf-z0, l, l ])
Xz_s = np.zeros([zf-z0, l, l ])

mascara_grosa = np.zeros( [l]*2 )
for j in range(l):
    for i in range(l):
        if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
            mascara_grosa[i,j] = 1

# mascara_grosa3 = mascara_grosa
for z in range(z0,zf,1):
    print(z)
    pre = stack_pre[z]
    post = correct_driff( stack_post[z+delta_z], pre, 20 )

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio

    Xz_s[z-z0] = X_s*mascara_grosa
    Yz_s[z-z0] = Y_s*mascara_grosa

    Xz[z-z0] = X_nmt*mascara_grosa
    Yz[z-z0] = Y_nmt*mascara_grosa
    
    scale0 = 100
    plt.figure()
    plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
    # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'Z = ' + str(z) )
    plt.xlim([0,1023])
    plt.ylim([1023,0])
    plt.show()
    
#%%

Rz3 = np.sqrt(Xz_s**2 + Yz_s**2)*ps_list[3]

#%%
Rz_dist = np.array( [ Rz[z].flatten()*ps for z in range( len(Rz) ) ] )
df = pd.DataFrame()
for i in range(zf-z0):
    df[ str((i - 5)/2) ] = Rz_dist[i]

#%%
plt.figure(figsize=[12,4])
sns.violinplot( df  )
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Profundidad [µm]" )
plt.grid()

#%%
# plt.hist( Rz_dist[2], bins=np.arange(-0.025, 1.025, 0.05) )
# z_plot = (np.arange(1.5, -2.5, -0.5 ))
z_plot = -0.5*(np.arange(z0, zf, 1 )-5)
Rz_mean = np.array( [ np.sum( Rz[z] )/np.sum( mascara_grosa ) for z in range( len(Rz) ) ] )*ps
# Rz_std = np.array( [ np.std( Rz[z] ) for z in range( len(Rz) -1) ] )*ps
# Rz_max = np.array( [ np.max( Rz[z] ) for z in range( len(Rz) -1) ] )*ps

plt.plot(z_plot, Rz_mean, 'o')
plt.xlim([1.7,-2.2])
# plt.plot(Rz_std)
# plt.plot(Rz_max)

#%%
ps_list = [0.0804, 0.0918, 0.1007, 0.1007]

#%%
Rz_dist0 = []
# ps = ps_list[0]

Rz_ = Rz0
mascara_grosa_ = mascara_grosa0
for k in range(Rz_.shape[0]):
    R_dist = []
    for j in range(Rz_.shape[1]):
        for i in range(Rz_.shape[2]):
            if mascara_grosa_[i,j] == 1:
                R_dist.append(Rz_[k,j,i])
    Rz_dist0.append(R_dist)
    
#%%

z_plot0 = -0.5*(np.arange(z0, len(Rz0)+z0, 1 )-5)
z_plot1 = -0.5*(np.arange(z0, len(Rz1)+z0, 1 )-5)
z_plot2 = -0.5*(np.arange(z0, len(Rz2)+z0, 1 )-5)
z_plot3 = -0.5*(np.arange(z0, len(Rz3)+z0, 1 )-5)

Rz_mean0 = np.array( [ np.sum( Rz0[z] )/np.sum( mascara_grosa0 ) for z in range( len(Rz0) ) ] )*ps_list[0]
Rz_mean1 = np.array( [ np.sum( Rz1[z] )/np.sum( mascara_grosa1 ) for z in range( len(Rz1) ) ] )*ps_list[1]
Rz_mean2 = np.array( [ np.sum( Rz2[z] )/np.sum( mascara_grosa2 ) for z in range( len(Rz2) ) ] )*ps_list[2]
Rz_mean3 = np.array( [ np.sum( Rz3[z] )/np.sum( mascara_grosa3 ) for z in range( len(Rz3) ) ] )*ps_list[3]

#%%
plt.figure(figsize=[12,8])
plt.plot(z_plot0, Rz_mean0, 'o', c = c1)
plt.plot(z_plot1[:-3], Rz_mean1[:-3], 'o', c = c2)
plt.plot(z_plot2[:-1], Rz_mean2[:-1], 'o', c = c3)
plt.plot(z_plot3, Rz_mean3, 'o', c = c4)
plt.ylabel( "Deformación promedio [µm]" )
plt.xlabel( "Profundidad [µm]" )
plt.grid(True)
plt.xlim([2,-4])

#%%
plt.figure(figsize=[12,4])
sns.violinplot( Rz_dist0, color = c1, cut = False)
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Profundidad [µm]" )
plt.xticks( np.arange(len(Rz_dist0)), z_plot0 )
# plt.xlim([12.5,-0.5])
plt.grid()

#%%
plt.figure(figsize=[12,4])
sns.violinplot( Rz_dist3, color = c4, cut = False)
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Profundidad [µm]" )
plt.xticks( np.arange(len(Rz_dist3)), z_plot3 )
# plt.xlim([12.5,-0.5])
plt.grid()






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


#%%
ps_lista = [0.0805,0.0914,0.1007,0.1007 ]

plt.figure( figsize = [12,5] )
plt.plot( np.arange( int(6/ps[0]), int(2/ps[0]),-1)*ps[0], Rz0, 'o' )
plt.plot( np.arange( int(6/ps[1]), int(2/ps[1]),-1)*ps[1], Rz1, 'o' )
plt.plot( np.arange( int(6/ps[2]), int(2/ps[2]),-1)*ps[2], Rz2, 'o' )
plt.plot( np.arange( int(6/ps[3]), int(2/ps[3]),-1)*ps[3], np.array(Rz3), 'o' )
plt.ylabel( "Deformación promedio [µm]" )
plt.xlabel( "Ventana final [µm]" )
plt.xlim([6.1,1.9])
plt.grid()

#%%

v0_int = np.arange( int(6/ps_lista[3]), int(2/ps_lista[3]),-1)
v_corta_int = [ v0_int[i] for i in range(len(v0)) if i%4 == 0 ]

#%%

Rz3_dist_corto = [ Rz3_dist[i] for i in range(len(Rz3_dist)) if i%4 == 0 ]

plt.figure(figsize=[12,4])
sns.violinplot( Rz3_dist_corto, color = (0.839, 0.152, 0.156))
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Ventana final [µm]" )
# plt.xticks( np.linspace(9, 0, 10 ) , np.round( np.linspace( int(6/ps_lista[3]), int(2/ps_lista[3]), 10 )*ps_lista[3] , 1) )
plt.xticks( np.linspace(-0.03, 9.01, 9), np.arange(6,1.5,-0.5)  )
# plt.xlim([12.5,-0.5])
plt.grid()

#%%

Rz0_dist_corto = [ Rz0_dist[i] for i in range(len(Rz0_dist)) if i%4 == 0 ]

plt.figure(figsize=[12,4])
sns.violinplot( Rz0_dist_corto, color = (0.122, 0.466, 0.705), cut = True )
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Ventana final [µm]" )
# plt.xticks( np.linspace(9, 0, 10 ) , np.round( np.linspace( int(6/ps_lista[3]), int(2/ps_lista[3]), 10 )*ps_lista[3] , 1) )
plt.xticks( np.linspace(-0.03, 12.01, 9), np.arange(6,1.5,-0.5)  )
plt.xlim([-0.7, 12.7])
plt.yticks([0,0.3,0.6,0.9,1.2])
plt.grid()
#%%
Rz3_dist_corto = [ Rz3_dist[i] for i in range(len(Rz3_dist)) if i%4 == 0 ]

plt.figure(figsize=[12,4])
sns.violinplot( Rz3_dist_corto, color = (0.839, 0.152, 0.156), cut = True )
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Ventana final [µm]" )
# plt.xticks( np.linspace(9, 0, 10 ) , np.round( np.linspace( int(6/ps_lista[3]), int(2/ps_lista[3]), 10 )*ps_lista[3] , 1) )
plt.xticks( np.linspace(-0.03, 9.01, 9), np.arange(6,1.5,-0.5)  )
plt.xlim([-0.5, 9.5])
plt.grid()
#%%
plt.plot( np.arange( int(6/ps_lista[3]), int(2/ps_lista[3]),-1)*ps_lista[3], np.array(Rz3), 'o' )
plt.plot( np.arange( int(6/ps_lista[3]), int(2/ps_lista[3]),-1)*ps_lista[3], [ np.mean( Rz3[i] ) for i in range(len(Rz3))  ]   )

#%%
Rz0_dist_corto = [ Rz0_dist[i] for i in range(len(Rz0_dist)) if i%6 == 0 ]
v0 = np.arange( int(6/ps_lista[0]), int(2/ps_lista[0]),-1)*ps_lista[0]
v_corta = [ v0[i] for i in range(len(v0)) if i%6 == 0 ]

plt.figure(figsize=[12,4])

for i in range(len(Rz3_dist_corto)):
    # print(i)
    dist, x = np.histogram( Rz0_dist_corto[i], bins = np.linspace(0, 2, 11), density=True )
    # plt.hist( i, bins = np.linspace(0, 2, 11) )
    dist = np.convolve(dist, [1/3]*3)
    plt.plot(x[:-1]+0.1, dist[1:-1], label = str(np.round( v_corta[i], 1 ) ) )
    # plt.legend()
plt.legend()
#%%

dist, x = np.histogram( Rz3_dist[0], bins = np.linspace(0, 2, 11) )
plt.plot(x[:-1]+0.1, dist)
#%%
plt.hist( Rz0_dist[-10], bins = np.arange(10)/10 )

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















#%%


cm1 = ListedColormap( [(1, 1, 1), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765) ] )
cm2 = ListedColormap( [(1, 1, 1), (1.0, 0.4980392156862745, 0.054901960784313725)] )
cm3 = ListedColormap( [(1, 1, 1), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)] )
cm4 = ListedColormap( [(1, 1, 1), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)] )
cm5 = ListedColormap( [(1, 1, 1), (0.5803921568627451, 0.403921568627451, 0.7411764705882353)] )

cm_list = [cm1,cm2,cm3,cm4]

#%% 4 celulas


plt.figure(figsize = [11, 11], tight_layout=True)

for cel in [0,1,2,3]:
 
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')
    
    pre = stack_pre[5]
    post, data = correct_driff_3D( stack_post, pre, 50, info = True )
    
    # Determinación del campo de deformación
    it = 3
    vi = int( int( 3/ps )*2**(it-1) )

    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio
    
    fs = 'large'
    plt.subplot(1,2,1)

    plt.imshow( celula_pre, cmap = 'gray' )
    plt.plot( b[1] ,b[0], c = 'w'  )
    barra_de_escala( 20, pixel_size = ps, img_len = 1000,  sep = 2,  font_size = fs, color = 'w' )

    plt.subplot(1,2,2)

    scale0 = 100
    plt.imshow( mascara, cmap = cm_list[cel], alpha = 0.5 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 20, pixel_size = ps, sep = 2,  font_size = fs, color = 'k' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])



#%%
plt.figure(figsize = [11, 11], tight_layout=True)
cel = 0
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = cm_list[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])


cel = 1
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = cm_list[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])



cel = 2
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(1,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = cm_list[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])



cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto.py')

pre = stack_pre[5]
post, data = correct_driff_3D( stack_post, pre, 50, info = True )

# Determinación del campo de deformación
it = 3
vi = int( int( 3/ps )*2**(it-1) )

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

plt.subplot(4,2,2*cel+1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w'  )
barra_de_escala( 10, pixel_size = ps, img_len = 1024,  sep = 3,  font_size = fs, color = 'w' )

plt.subplot(4,2,2*cel+2)

scale0 = 100
plt.imshow( mascara, cmap = cm_list[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, pixel_size = ps, sep = 3,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])













