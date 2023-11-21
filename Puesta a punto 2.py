# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:26:49 2023

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
cm_ar = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (0.839, 0.152, 0.157)] ) 
cm_aa = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (1.000, 0.498, 0.055)] ) 
cm_aa2 = ListedColormap( [(0.122, 0.467, 0.706), (0, 0, 0), (1.000, 0.498, 0.055)] ) 

c0 = (0.122, 0.467, 0.706)
c1 = (1.000, 0.498, 0.055)
c2 = (0.173, 0.627, 0.173)
c3 = (0.839, 0.152, 0.157)
colores = [c0, c1, c2, c3]

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )
color_maps = [cm0, cm1, cm2, cm3]

#MCF7 D30 R4 cel9 del 1/9   0
#MCF7 C30 R5 cel5 del 1/9   1
#MCF10 D04 R9 5/10          2
#MCF10 G18 R25 del 19/10    3
#%%
cel = 0

#%% Invocacion
path = r"C:\Users\gonza\1\Tesis\2023\\"
nombres = [ 'MCF7 D30_R04', 'MCF7 C30_R05', 'MCF10 D04_R09', 'MCF10 G18_R25'  ]
regiones = [ 4, 5, 9, 25 ]
img_trans = [ 0, 0, 2, 2 ]
As = [ 0.85, 0.8, 0.8, 0.75]
ps_list = [0.0804, 0.0918, 0.1007, 0.1007]

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
cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
#%% Dependencia Axial

Rz_dist_todas = []
z0 = 2
color_maps = [cm0, cm1, cm2, cm3]

for iterador in range(4):
    cel = iterador
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Axial.py')
    
    it = 3
    vi = int( int( np.round(3/ps) )*2**(it-1) )
    bordes_extra = 8
    
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 2.5
    modo = "Smooth3"
    suave0 = 3
    fs = 'x-large'

    zf = min( len(stack_pre), len(stack_post) ) - 1

    pre = stack_pre[5]
    post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )
    delta_z = ZYX[0] - 5

    Rz_dist_celula = []
    
    for z in range(z0,zf,1):
        print(z)
        pre = stack_pre[z]
        post = correct_driff( stack_post[ z + delta_z ], pre, 20 )
    
        dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
        Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
        X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
        x, y = dominio
        l = len(x)
    
        # Rz_ = np.sqrt( X_nmt**2 + Y_nmt**2 )
        Rz_ = np.sqrt( X_s**2 + Y_s**2 )
        Rz_dist_ = []
        for j in range(l):
            for i in range(l):
                if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                    Rz_dist_.append( Rz_[j,i]*ps )

        scale0 = 100
        plt.figure()
        plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
        # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
        plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
        barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'Z = ' + str(z) )
        plt.xlim([0,1023])
        plt.ylim([1023,0])
        plt.show()

        Rz_dist_celula.append(Rz_dist_)
    
    Rz_dist_todas.append(Rz_dist_celula)

Rz_dist_todas[3].pop(-1)
#%% 

cel = 3
data = Rz_dist_todas[cel]

plt.figure(figsize=[12,4])
sns.violinplot( data, color =  colores[cel], cut = False )
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Profundidad [µm]" )
plt.xticks( np.arange(len(data)), -np.arange(len(data))/2 )
plt.grid()

#%%

plt.figure(figsize=[10,10], tight_layout = True)

for c in range(4):
    cel = c
    
    data = Rz_dist_todas[cel]
    plt.subplot(4,1,c+1)

    sns.violinplot( data, color =  colores[cel], cut = False )
    plt.ylabel( "Deformación [µm]" )
    plt.grid()
    # plt.xticks( np.arange(len(data)) )   
    plt.yticks([0.0,0.3,0.6,0.9],[0.0,0.3,0.6,0.9])
    plt.xticks( ticks = np.arange(len(data)), labels=[] )   
    plt.xlim([-0.5,9.5])
    if c == 3:
        plt.xlabel( "Profundidad [µm]" )
        plt.xticks( np.arange(len(data)), -np.arange(len(data))/2 )   
        # plt.xlim([0.25,-3.25])
        plt.yticks([0.0,0.3,0.6,0.9,1.2,1.5],[0.0,0.3,0.6,0.9,1.2,1.5])
        plt.xlim([-0.5,9.5])

#%%

promRz0 = np.array( [ np.mean(Rz_dist_todas[0][i]) for i in range(len( Rz_dist_todas[0]) ) ]  ) 
promRz1 = np.array( [ np.mean(Rz_dist_todas[1][i]) for i in range(len( Rz_dist_todas[1]) ) ]  ) 
promRz2 = np.array( [ np.mean(Rz_dist_todas[2][i]) for i in range(len( Rz_dist_todas[2]) ) ]  ) 
promRz3 = np.array( [ np.mean(Rz_dist_todas[3][i]) for i in range(len( Rz_dist_todas[3]) ) ]  ) 
promRz = [promRz0, promRz1, promRz2, promRz3 ] 

#%%
plt.figure( figsize = [10,7] )
plt.plot( np.arange( len(promRz0) )/2 , promRz0, 'o', color = colores[0] )
plt.plot( np.arange( len(promRz1) )/2, promRz1, 'o', color = colores[1] )
plt.plot( np.arange( len(promRz2) )/2, promRz2, 'o', color = colores[2] )
plt.plot( np.arange( len(promRz3) )/2, promRz3, 'o', color = colores[3] )
plt.ylabel( "Def. promedio [µm]" )
plt.xlabel( "Profundidad [µm]" )
# plt.xlim([5.4,1.6])
plt.grid()












#%%

Rv_dist_todas = []
z0 = 5
color_maps = [cm0, cm1, cm2, cm3]


for cel in range(4):
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')    
    ventanas = np.arange( int(5/ps), int(2/ps),-1)
    
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 2.5
    modo = "Smooth3"
    suave0 = 3
    fs = 'x-large'

    zf = min( len(stack_pre), len(stack_post) ) - 1

    pre = stack_pre[z0]
    post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )

    Rv_dist_celula = []
    
    for vf in ventanas:
        print(vf)
    
        it = 3
        vi = int( int( vf )*2**(it-1) )
        bordes_extra = 8
    
        dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
        Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
        X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
        x, y = dominio
        l = len(x)
    
        Rv_ = np.sqrt( X_s**2 + Y_s**2 )
        Rv_dist_ = []    
        for j in range(l):
            for i in range(l):
                if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                    Rv_dist_.append( Rv_[j,i]*ps )

        scale0 = 100
        plt.figure()
        plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
        # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
        plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
        barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k', more_text = 'V = ' + str(vf) )
        plt.xlim([0,1023])
        plt.ylim([1023,0])
        plt.show()

        Rv_dist_celula.append(Rv_dist_)
    
    Rv_dist_todas.append(Rv_dist_celula)

#%% 

ventanas0 = np.arange( int(5/ps_list[0]), int(2/ps_list[0]),-1)
ventanas1 = np.arange( int(5/ps_list[1]), int(2/ps_list[1]),-1)
ventanas2 = np.arange( int(5/ps_list[2]), int(2/ps_list[2]),-1)
ventanas3 = np.arange( int(5/ps_list[3]), int(2/ps_list[3]),-1)
ventanas = [ventanas0,ventanas1,ventanas2,ventanas3]

promR0 = np.array( [ np.mean(Rv_dist_todas[0][i]) for i in range(len( Rv_dist_todas[0]) ) ]  ) 
promR1 = np.array( [ np.mean(Rv_dist_todas[1][i]) for i in range(len( Rv_dist_todas[1]) ) ]  ) 
promR2 = np.array( [ np.mean(Rv_dist_todas[2][i]) for i in range(len( Rv_dist_todas[2]) ) ]  ) 
promR3 = np.array( [ np.mean(Rv_dist_todas[3][i]) for i in range(len( Rv_dist_todas[3]) ) ]  ) 
promR = [promR0[0], promR1[1], promR2[2], promR3[3] ] 
#%%



#%%
cel = 3
data = Rv_dist_todas[cel]

data_plot = []
ventanas_plot = []
for i in range(len(ventanas[cel])):
    if i%4 == 0:
        data_plot.append( data[i] )
        ventanas_plot.append( ventanas[cel][i] )

plt.figure(figsize=[12,4])
sns.violinplot( data_plot, color =  colores[cel], split = False, cut = False)
plt.ylabel( "Deformación [µm]" )
plt.xlabel( "Profundidad [µm]" )
m = (  len(data_plot)-1   )/(  (np.array(ventanas_plot)*ps_list[cel])[-1] - (np.array(ventanas_plot)*ps_list[cel])[0]  )
b = -m*(np.array(ventanas_plot)*ps_list[cel])[0]
plt.xticks( m*np.arange(5,1.5,-0.5)+b, np.arange(5,1.5,-0.5) )
plt.xlim( [m*5.5+b, m*1.6+b ] )
plt.grid()

#%%

plt.figure( figsize = [10,7] )
plt.plot( ventanas0*ps_list[0], promR0, 'o', color = colores[0] )
plt.plot( ventanas1*ps_list[1], promR1, 'o', color = colores[1] )
plt.plot( ventanas2*ps_list[2], promR2, 'o', color = colores[2] )
plt.plot( ventanas3*ps_list[3], promR3, 'o', color = colores[3] )
plt.ylabel( "Def. promedio [µm]" )
plt.xlabel( "Ventana final [µm]" )
plt.xlim([5.4,1.6])
plt.grid()

#%%

#%%
plt.figure(figsize=[10,10], tight_layout = True)

for c in range(4):
    cel = c
    data = Rv_dist_todas[cel]
    data_plot = []
    ventanas_plot = []
    for i in range(len(ventanas[cel])):
        if i%4 == 0:
            data_plot.append( data[i] )
            ventanas_plot.append( ventanas[cel][i] )
    
    plt.subplot(4,1,c+1)
    sns.violinplot( data_plot, color =  colores[cel], split = False, cut = False)
    plt.ylabel( "Deformación [µm]" )
    m = (  len(data_plot)-1   )/(  (np.array(ventanas_plot)*ps_list[cel])[-1] - (np.array(ventanas_plot)*ps_list[cel])[0]  )
    b = -m*(np.array(ventanas_plot)*ps_list[cel])[0]
    # plt.xticks( m*np.arange(5,1.5,-0.5)+b, np.arange(5,1.5,-0.5) )
    plt.xlim( [m*5.5+b, m*1.6+b ] )
    plt.grid()
    if c != 3:
        plt.xticks( m*np.arange(5,1.5,-0.5)+b )
        plt.yticks([0.0,0.3,0.6,0.9],[0.0,0.3,0.6,0.9])
        plt.ylim(-0.035, 0.935)
    if c == 3:
        plt.xlabel( "Ventana final [µm]" )
        plt.xticks( m*np.arange(5,1.5,-0.5)+b, np.arange(5,1.5,-0.5) )    
    if c == 0:
        plt.yticks([0.0,0.3,0.6,0.9],[0.0,0.3,0.6,0.9])

# plt.subplot(5,1,5, subplotspec = 2)
# plt.plot( ventanas0*ps_list[0], promR0, 'o', color = colores[0] )
# plt.plot( ventanas1*ps_list[1], promR1, 'o', color = colores[1] )
# plt.plot( ventanas2*ps_list[2], promR2, 'o', color = colores[2] )
# plt.plot( ventanas3*ps_list[3], promR3, 'o', color = colores[3] )
# plt.ylabel( "Def. promedio [µm]" )
# plt.xlabel( "Ventana final [µm]" )
# plt.xlim([5.4,1.6])
# plt.grid()

#%%























#%%
cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
#%% Deformación Axial

domX = []
domY = []
mapasX = []
mapasY = []
axial = []
mascaras = []

for iterador in range(4):
    cel = iterador
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
    
    it = 3
    vi = int( int( 3/ps )*2**(it-1) )
    bordes_extra = int(np.round(vi/2**(it-1))/4) 
    
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 2.5
    modo = "Smooth3"
    suave0 = 3
    
    pre = stack_pre[5]
    post, ZYX = correct_driff_3D( stack_post, pre, 40, info = True )
    
    control0 = [(-1,-1)]
    dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel], control = control0 )
    Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
    X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
    x, y = dominio

    z_0 = ZYX[0] - 5
    # if cel == 2:
    #     z_0 = 0
    # if cel == 1:
    #     z_0 = 0    
    # z_0 = 0
    Z = Z_iteration0( stack_post0 = stack_post, img_pre0 = pre, win_shape = int( 3/ps ), exploration = 1, translation_Y = Y_nmt, translation_X = X_nmt, mode = "Smooth3", A = As[cel], z0 = z_0 )

    domX.append(x)
    domY.append(y)
    mapasX.append(X_s)    
    mapasY.append(Y_s)
    axial.append(Z)
    mascaras.append(mascara)

#%%

plt.figure( figsize=[10,10], tight_layout = True )
fs = 'x-small'
for cel in range(4):

    ps = ps_list[cel]
    x = domX[cel]
    y = domY[cel]
    X_s = mapasX[cel]    
    Y_s = mapasY[cel]
    Z = axial[cel]
    mascara = mascaras[cel]

    plt.subplot(4, 3, 3*cel+1 )
    scale0 = 100
    plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'k' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    R = np.sqrt( X_s**2 + Y_s**2 )*ps
    plt.subplot(4, 3, 3*cel+2 )
    plt.imshow( R, vmin = 0, vmax = np.max(R), extent = [x[0,0]-int( 3/ps )/2, x[-1,-1]+int( 3/ps )/2,x[-1,-1]+int( 3/ps )/2,x[0,0]-int( 3/ps )/2] )
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'w' )

    plt.subplot(4, 3, 3*cel+3 )
    plt.imshow( Z , extent = [x[0,0]-int( 3/ps )/2, x[-1,-1]+int( 3/ps )/2,x[-1,-1]+int( 3/ps )/2,x[0,0]-int( 3/ps )/2])
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'w' )    


#%%
plt.figure( figsize=[12,9], tight_layout = True )
fs = 'medium'
ubicaciones = [1,5,9,2,6,10,3,7,11,4,8,12]
for cel in range(4):

    ps = ps_list[cel]
    x = domX[cel]
    y = domY[cel]
    X_s = mapasX[cel]    
    Y_s = mapasY[cel]
    Z = axial[cel]
    mascara = mascaras[cel]

    plt.subplot(3, 4, ubicaciones[3*cel] )
    scale0 = 100
    plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
    plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'k' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    R = np.sqrt( X_s**2 + Y_s**2 )*ps
    plt.subplot(3, 4, ubicaciones[3*cel+1] )
    plt.imshow( R, vmin = 0, vmax = np.max(R), extent = [x[0,0]-int( 3/ps )/2, x[-1,-1]+int( 3/ps )/2,x[-1,-1]+int( 3/ps )/2,x[0,0]-int( 3/ps )/2] )
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'w', text = False )

    plt.subplot(3, 4, ubicaciones[3*cel+2] )
    plt.imshow( Z , extent = [x[0,0]-int( 3/ps )/2, x[-1,-1]+int( 3/ps )/2,x[-1,-1]+int( 3/ps )/2,x[0,0]-int( 3/ps )/2])
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'w', text = False )    








#%%


it = 3
vi = int( int( 3/ps )*2**(it-1) )
bordes_extra = int(np.round(vi/2**(it-1))/4) 

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
# modo = "Fit"
suave0 = 3

pre = stack_pre[5]
post, ZYX = correct_driff_3D( stack_post, pre, 40, info = True )

j_p, i_p = np.random.randint( len(x) - 2 ) + 1, np.random.randint( len(x) - 2 ) + 1
control0 = [(j_p,i_p)]
# control0 = [(-1,-1)]

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])#, control = control0 )
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio

#%%
scale0 = 100
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X_nmt**2 + Y_nmt**2) )*ps, 2)  )

#%%
scale0 = 100
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X_s**2 + Y_s**2) )*ps, 2)  )

#%%
z_0 = ZYX[0] - 5
Z = Z_iteration0( stack_post0 = stack_post, img_pre0 = pre, win_shape = int( 3/ps ), exploration = 1, translation_Y = Y_nmt, translation_X = X_nmt, mode = "Smooth3", A = As[cel], z0 = z_0 )
j_p, i_p = np.random.randint( len(x) - 2 ) + 1, np.random.randint( len(x) - 2 ) + 1
# j_p, i_p = -1, -1

#%%
plt.figure()
plt.imshow(Z, cm_aa)
plt.colorbar(ticks = [-1,0,1])
barra_de_escala( 10, img_len = len(Z)-1, pixel_size = ps_list[cel]*1024/len(Z), sep = 1.5,  font_size = fs, color = 'k' )

#%%
scale0 = 100
# plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.2 )
# plt.quiver(x,y,X_s,-Y_s, Z, cmap = cm_aa2 ,scale = scale0, pivot='tail')
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
plt.imshow(Z, cm_aa, alpha = 0.4, extent = [x[0,0]-int( 3/ps )/2, x[-1,-1]+int( 3/ps )/2,x[-1,-1]+int( 3/ps )/2,x[0,0]-int( 3/ps )/2])
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )

plt.xlim([0,1023])
plt.ylim([1023,0])

#%%

scale0 = 100
plt.imshow( mascara, cmap = cm_y, alpha = 0.5 )
plt.quiver(x,y,X + 1,-Y-1, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])

print( np.round( np.max( np.sqrt(X**2 + Y**2) )*ps, 2)  )









































