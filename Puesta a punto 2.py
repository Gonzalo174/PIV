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
plt.rcParams['figure.figsize'] = [6,6]
plt.rcParams['font.size'] = 11
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
cel = 1

#%% Invocacion
l = 6
# path = r"D:\\Gonzalo\\"
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
    mascara =  1 - np.loadtxt( path[:-l] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_00um.png")
    mascara10 =  1 - np.loadtxt( path[:-l] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_10um.png")
    mascara20 =  1 - np.loadtxt( path[:-l] + r"PIV\Mascaras MCF7\\" + nombres[cel][-7:] + "_m_20um.png")

elif cel == 2 or cel ==3:
    mascara =  np.loadtxt( path[:-l] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_00um.csv")
    mascara10 =  np.loadtxt( path[:-l] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_10um.csv")
    mascara20 =  np.loadtxt( path[:-l] + r"PIV\Mascaras MCF10\\" + nombres[cel][-7:] + "_m_20um.csv")

b = border(mascara, 600)








#%% Seleccion de ventanas

A = As[cel]
ws = 2.5
pre = stack_pre[5]
desvios_bin, limit = busca_esferas( pre, ps = ps, th = A, win = ws )
plt.figure( figsize = [6, 6] )
plt.imshow( pre[ :limit , :limit ], cmap = cm_crimson, vmin = 150, vmax = 600 )
plt.imshow( 1-desvios_bin, cmap = 'gray', alpha = 0.09, extent = [0,limit,limit,0])
barra_de_escala( 10, sep = 2, more_text = 'Célula ' + str(cel), a_lot_of_text = str( int( np.mean( desvios_bin )*100 ) ) + '%', font_size = 'large' )
print(  np.mean(desvios_bin) )

#%%
mascara_aptitud = np.zeros([limit]*2)
aptas = []
no_aptas = []

for j in range(limit):
    for i in range(limit):
        if desvios_bin[ j//int(ws/ps), i//int(ws/ps) ] == 1:
            mascara_aptitud[j,i] = 1
            aptas.append( pre[j,i] )
        else:
            no_aptas.append( pre[j,i] )

plt.imshow(mascara_aptitud)

#%%
plt.title("Todas")
plt.hist(aptas+no_aptas, bins = np.arange(90, 610, 1), density=True)
plt.ylim([0,0.02])
plt.show()

#%%

b = np.arange(90, 610, 4)
todas_dist = np.histogram( pre.flatten(), bins = b, density=True  )
aptas_dist = np.histogram( aptas, bins = b, density=True  )
no_aptas_dist = np.histogram( no_aptas, bins = b, density=True  )
#%%
plt.plot( b[:-1], todas_dist[0], label = "Imagen completa" )
plt.plot( b[:-1], aptas_dist[0], label = "Ventanas aptas" )
plt.plot( b[:-1], no_aptas_dist[0], label = "Ventanas no aptas" )
plt.legend()
plt.grid()

#%%
plt.figure(figsize = [6,4])
plt.plot( b[:-1], normalizar(todas_dist[0]), label = "Imagen completa" )
plt.plot( b[:-1], normalizar(aptas_dist[0]), label = "Ventanas aptas" )
plt.plot( b[:-1], normalizar(no_aptas_dist[0]), label = "Ventanas no aptas" )
plt.legend()
plt.grid()

#%%
cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
# runcell('Invocacion', 'D:/Gonzalo/PIV/Puesta a punto 2.py')

ws = 2.5
it = 3
vi = int( int( np.round(ws/ps) )*2**(it-1) )
bordes_extra = 8

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Smooth3"
suave0 = 3

pre = stack_pre[5]
post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )
#%%

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio
print(  np.mean(res) )

#%%
fs = 'small'

scale0 = 100
plt.figure(figsize = [6,4] )
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
# plt.quiver(x,y,X_0,-Y_0, res, cmap = cm_crimson, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.show()

#%% Plot NMT

fs = 'small'

scale0 = 50
plt.figure(figsize = [6,4] )
plt.imshow( smooth(mascara,12), cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_0,-Y_0, res, width = 0.008, cmap = cm_crimson, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, width = 0.002, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
# barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([200 + ws/ps/2,400 - ws/ps/2])
plt.ylim([400 - ws/ps/2,200 + ws/ps/2])
plt.show()

#%%
fondo_pre = median_blur(celula_pre, 50 )
#%%
# mascara = smooth(mascara, 11)
# b = border(mascara, 600)
fs = 'large'

plt.figure(figsize = [7,7], layout = 'compressed')
a = np.mean( pre )/np.mean( post )
a_x, b_x = 225, 375
a_y, b_y = 225, 375

xo, yo = 257, 257

plt.subplot(1,3,1)
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 400 )
plt.imshow( post, cmap = cm_green, vmin = 150, vmax = 400/(a + 0.1), alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
# barra_de_escala( 10, pixel_size = ps, img_len = np.abs(a_x - b_x), sep = 0.8,  font_size = fs, color = 'w', text = False)
plt.xlim([a_x,b_x])
plt.ylim([b_y,a_y])
plt.xticks([])
plt.yticks([])
for i in np.arange(-5, -10, -0.1):
    plt.plot([ xo-25, xo ], [yo + i, yo + i], color='w', linewidth = 2)
plt.text( xo - 12.5, yo - 14 , '2 µm', color= 'w', weight='bold', ha='center', va = 'bottom', fontsize = fs )


plt.subplot(1,3,2)
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.9 )
plt.quiver(x,y,X_0,-Y_0, res, width = 0.008, cmap = cm_crimson, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_s,-Y_s, width = 0.006, scale = 100,  pivot='tail')
# barra_de_escala( 10, sep = 0.8, img_len = np.abs(a_x - b_x),  font_size = fs, color = 'k', text = False)
plt.xlim([a_x,b_x])
plt.ylim([b_y,a_y])
plt.xticks([])
plt.yticks([])
for i in np.arange(-5, -10, -0.1):
    plt.plot([ xo-25, xo ], [yo + i, yo + i], color='k', linewidth = 2)
plt.text( xo - 12.5, yo - 14 , '2 µm', color= 'k', weight='bold', ha='center', va = 'bottom', fontsize = fs )
plt.text( (b_x+a_x)/2, b_y+20 , 'Antes del NMT', color= 'k', weight='bold', ha='center', va = 'bottom', fontsize = 14)


plt.subplot(1,3,3)
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.9 )
# plt.quiver(x,y,X_0,-Y_0, res, width = 0.008, cmap = cm_crimson, scale = scale0, pivot='tail')
plt.quiver(x,y,X_nmt,-Y_nmt, res, width = 0.008, cmap = cm_crimson, scale = scale0,  pivot='tail')
# barra_de_escala( 10, sep = 0.8, img_len = np.abs(a_x - b_x),  font_size = fs, color = 'k', text = False)
plt.xlim([a_x,b_x])
plt.ylim([b_y,a_y])
plt.xticks([])
plt.yticks([])
for i in np.arange(-5, -10, -0.1):
    plt.plot([ xo-25, xo ], [yo + i, yo + i], color='k', linewidth = 2)
plt.text( xo - 12.5, yo - 14 , '2 µm', color= 'k', weight='bold', ha='center', va = 'bottom', fontsize = fs )
plt.text( (b_x+a_x)/2, b_y+20 , 'Después del NMT', color= 'k', weight='bold', ha='center', va = 'bottom', fontsize = 14 )


plt.show()












#%%
cel = 3
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
#%% Dependencia Axial

Rz_dist_todas = []
cobertura_todas = []
D_cobertura_todas = []

z0 = 2

for iterador in range(4):
    cel = iterador
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
    
    it = 3
    vi = int( int( np.round(3/ps) )*2**(it-1) )
    bordes_extra = 8
    
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 7.5
    modo = "Smooth3"
    suave0 = 3
    fs = 'small'

    zf = min( len(stack_pre), len(stack_post) ) - 1

    pre = stack_pre[5]
    post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )
    delta_z = ZYX[0] - 5
    desvio0 = np.std( pre )

    Rz_dist_celula = []
    cobertura_celula = []
    D_cobertura_celula = []
    
    for z in range(z0,zf,1):
        print(z)
        pre = stack_pre[z]
        post = correct_driff( stack_post[ z + delta_z ], pre, 20 )
    
        des, lim = busca_esferas( pre, ps = ps_list[cel], win = 2, th = As[cel], std_img = desvio0 )
        des_a, lim = busca_esferas( pre, ps = ps_list[cel], win = 2, th = As[cel] + 0.05*As[cel], std_img = desvio0 )
        des_b, lim = busca_esferas( pre, ps = ps_list[cel], win = 2, th = As[cel] - 0.05*As[cel], std_img = desvio0 )
    
        dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
        Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
        X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
        x, y = dominio
        l = len(x)
    
        Rz_ = np.sqrt( X_nmt**2 + Y_nmt**2 )
        # Rz_ = np.sqrt( X_s**2 + Y_s**2 )
        Rz_dist_ = []
        for j in range(l):
            for i in range(l):
                if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                    Rz_dist_.append( Rz_[j,i]*ps )

        scale0 = 100
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
        # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
        # plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
        plt.quiver(x,y,deformacion[1],-deformacion[0], res, cmap = cm_crimson, scale = scale0, pivot='tail')
        barra_de_escala( 10, sep = 1.5, pixel_size = ps_list[cel], font_size = fs, color = 'k', more_text = 'Z = ' + str(z) )
        plt.xlim([0,1023])
        plt.ylim([1023,0])
        
        plt.subplot(1,2,2)
        plt.imshow(pre, cmap = cm_crimson, vmin = 100, vmax = 600)
        barra_de_escala( 10, sep = 1.5, pixel_size = ps_list[cel], font_size = fs, color = 'w', more_text = 'Z = ' + str(z) )
        plt.show()

        Rz_dist_celula.append(Rz_dist_)
        cobertura_celula.append( np.mean(des) ) 
        D_cobertura_celula.append( np.max( [ np.abs(np.mean(des) - np.mean(des_a)), np.abs(np.mean(des) - np.mean(des_b)) ] ) ) 

    Rz_dist_todas.append(Rz_dist_celula)
    cobertura_todas.append(cobertura_celula)
    D_cobertura_todas.append(D_cobertura_celula)

Rz_dist_todas[3].pop(-1)

#%%

promRz0 = np.array( [ np.mean(Rz_dist_todas[0][i]) for i in range(len( Rz_dist_todas[0]) ) ]  ) 
promRz1 = np.array( [ np.mean(Rz_dist_todas[1][i]) for i in range(len( Rz_dist_todas[1]) ) ]  ) 
promRz2 = np.array( [ np.mean(Rz_dist_todas[2][i]) for i in range(len( Rz_dist_todas[2]) ) ]  ) 
promRz3 = np.array( [ np.mean(Rz_dist_todas[3][i]) for i in range(len( Rz_dist_todas[3]) ) ]  ) 
promRz = [promRz0, promRz1, promRz2, promRz3 ] 

stdRz0 = np.array( [ np.std(Rz_dist_todas[0][i]) for i in range(len( Rz_dist_todas[0]) ) ]  ) 
stdRz1 = np.array( [ np.std(Rz_dist_todas[1][i]) for i in range(len( Rz_dist_todas[1]) ) ]  ) 
stdRz2 = np.array( [ np.std(Rz_dist_todas[2][i]) for i in range(len( Rz_dist_todas[2]) ) ]  ) 
stdRz3 = np.array( [ np.std(Rz_dist_todas[3][i]) for i in range(len( Rz_dist_todas[3]) ) ]  ) 
stdRz = [stdRz0, stdRz1, stdRz2, stdRz3 ] 


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

plt.rcParams['figure.figsize'] = [6,3]
plt.rcParams['font.size'] = 10

cel = 3

fig, ax1 = plt.subplots()

line1, = ax1.plot(np.arange( len(promRz[cel]) )/2 , promRz[cel], 'o', color = colores[cel], label = "Deformación" )
ax1.set_ylabel('Deformación promedio [µm]')
ax1.set_xlabel('Profundidad [µm]', color='k')
ax1.set_ylim([0.360001,0.49999])

ax2 = ax1.twinx()
line2, = ax2.plot(np.arange( len(cobertura_todas[cel]) )/2, cobertura_todas[cel], 'o' , color='k', label='Cobertura')
ax2.set_ylabel('Fracción del plano cubierta', color='k')
ax2.set_ylim([-0.05,1.05])
# ax2.tick_params('y', colors='r')

# Añadir leyendas y mostrar el gráfico
ax1.grid()
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')
# ax1.legend(loc = 'lower right')
# ax2.legend(loc = 'upper right')
fig.tight_layout()
plt.show()

#%%

plt.rcParams['font.size'] = 12
fig, axs = plt.subplots(4, 1, figsize=(7, 11))
orden = [3,2,1,0]

# Trazar en cada subgráfico
for iterador, ax in enumerate(axs):

    cel = orden[iterador]
    
    # line1, = ax.plot(np.arange( len(promRz[cel]) )/2 , promRz[cel], 'o', color = colores[cel], label = "Deformación" )
    line1 = ax.errorbar(np.arange( len(promRz[cel]) )/2 , promRz[cel], stdRz[cel]/4, fmt = 'o', color = colores[cel], label = "Deformación" )
    ax.set_ylabel('Deformación promedio [µm]')
    
    if iterador < 3:
        # ax.set_ylim([0.09,0.21])
        ax.set_xticks(np.arange(7),['']*7)
    else:
        # ax.set_ylim([0.35,0.5])
        ax.set_xlabel('Profundidad [µm]', color='k')
    ax.set_xlim([-0.2,6.2])
    
    
    ax2 = ax.twinx()
    # line2, = ax2.plot(np.arange( len(cobertura_todas[cel]) )/2, cobertura_todas[cel], 'o' , color='k', label='Cobertura')
    line2 = ax2.errorbar(np.arange( len(cobertura_todas[cel]) )/2, cobertura_todas[cel], D_cobertura_todas[cel], fmt = 'o' , color='k', label='Cobertura')
    ax2.set_ylabel('Fracción del plano cubierta', color='k')
    ax2.set_ylim([-0.19,1.19])
    
    # Añadir leyendas y mostrar el gráfico
    ax.grid()
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right')
    
fig.tight_layout()
    
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Crear la figura y los subgráficos
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Trazar en cada subgráfico
for i, ax in enumerate(axs):
    # Subgráfico principal (izquierda)
    line1, = ax.plot(x, y1, color='b', label='y1')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('y1', color='b')
    ax.tick_params('y', colors='b')

    # Eje y adicional (derecha)
    ax2 = ax.twinx()
    line2, = ax2.plot(x, y2, color='r', label='y2')
    ax2.set_ylabel('y2', color='r')
    ax2.tick_params('y', colors='r')

    # Añadir leyenda combinada en el eje izquierdo
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper left')

# Ajustar espaciado entre subgráficos
plt.tight_layout()

# Añadir título general y mostrar el gráfico
fig.suptitle('Subgráficos Apilados con Ejes y Adicionales', fontsize=14)
plt.show()


#%%
# Datos de ejemplo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Crear la figura y el primer eje
fig, ax1 = plt.subplots()

# Trazar en el primer eje (izquierda)
ax1.plot(x, y1, color='b', label='y1')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('y1', color='b')
ax1.tick_params('y', colors='b')

# Crear el segundo eje y
ax2 = ax1.twinx()

# Trazar en el segundo eje (derecha)
ax2.plot(x, y2, color='r', label='y2')
ax2.set_ylabel('y2', color='r')
ax2.tick_params('y', colors='r')

# Añadir leyendas y mostrar el gráfico
fig.tight_layout()
plt.show()
















#%%

cel = 3
promRz0 = np.array( [ np.mean(Rz_dist_todas[cel][i]) for i in range(len( Rz_dist_todas[cel]) ) ]  ) 
stdRz0 = np.array( [ np.std(Rz_dist_todas[cel][i]) for i in range(len( Rz_dist_todas[cel]) ) ]  ) 


plt.errorbar(  np.arange( len(cobertura_todas[cel]) )/2, cobertura_todas[cel], D_cobertura_todas[cel], fmt = 'o', c = 'k')
plt.errorbar(  np.arange( len(cobertura_todas[cel]) )/2, promRz0/0.2, stdRz0/4, fmt = 'o', c = colores[cel])


#%%
cel = 0
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

cobertura_todas = []
D_cobertura_todas = []
z0 = 2

for iterador in range(1):
    cel = iterador
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
    
    pre0 = stack_pre[5]
    desvio0 = np.std( pre0 )
    cobertura_celula = []
    D_cobertura_celula = []
    
    for z in range(z0,zf,1):
        print(z)
        pre = stack_pre[z]
        des, lim = busca_esferas( pre, ps = ps_list[cel], win = 2, th = As[cel], std_img = desvio0 )
        des_a, lim = busca_esferas( pre, ps = ps_list[cel], win = 2, th = As[cel] + 0.05*As[cel], std_img = desvio0 )
        des_b, lim = busca_esferas( pre, ps = ps_list[cel], win = 2, th = As[cel] - 0.05*As[cel], std_img = desvio0 )
        
        
        cobertura_celula.append( np.mean(des) ) 
        D_cobertura_celula.append( np.max( [ np.abs(np.mean(des) - np.mean(des_a)), np.abs(np.mean(des) - np.mean(des_b)) ] ) ) 

        
    cobertura_todas.append(cobertura_celula)
    D_cobertura_todas.append(D_cobertura_celula)

#%%
# plt.plot(cobertura_todas[0], 'o')
plt.errorbar(  np.arange( len(cobertura_todas[0]) )/2, cobertura_todas[0], D_cobertura_todas[0], fmt = 'o', c = 'k')






















#%%
A = As[cel]
ws = 1
pre = stack_pre[5]
desvios_bin, limit = busca_esferas( pre, ps = ps, th = A, win = ws )
plt.figure( figsize = [6, 6] )
plt.imshow( pre[ :limit , :limit ], cmap = cm_crimson, vmin = 150, vmax = 600 )
plt.imshow( desvios_bin, cmap = 'gray', alpha = 0.09, extent = [0,limit,limit,0])
barra_de_escala( 10, sep = 2, more_text = 'Célula ' + str(cel), a_lot_of_text = str( int( np.mean( desvios_bin )*100 ) ) + '%', font_size = 'large' )
print(  np.mean(desvios_bin) )





# cuantifiacar el NMT multiplicado por el modulo los vectores que elimina
#%% Ventanas de exploracin

v_a, v_b = 5, 1
Rv_dist_todas = []
cobertura_todas = []
NMT_list_todas = []
# angulo_todas = []
z0 = 5
color_maps = [cm0, cm1, cm2, cm3]

for cel in range(4):
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')    
    ventanas = np.arange( int(v_a/ps), int(v_b/ps),-1)
    
    Noise_for_NMT = 0.2
    Threshold_for_NMT = 7.5
    modo = "Smooth3"
    suave0 = 3
    fs = 'small'

    zf = min( len(stack_pre), len(stack_post) ) - 1

    pre = stack_pre[z0]
    post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )

    Rv_dist_celula = []
    cobertura_celula = []
    NMT_list_celula = []
    # angulo_celula = []
    
    for vf in ventanas:
        print(vf)
    
        it = 3
        vi = int( int( vf )*2**(it-1) )
        bordes_extra = 8
        # it = 1
    
        desvios_bin, limit = busca_esferas( pre, ps = ps_list[cel], th = As[cel], win = vf*ps )
        dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel])
        Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
        X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
        x, y = dominio
        l = len(x)
    
        # CM = center_of_mass(mascara)
        
        Rv_ = np.sqrt( X_nmt**2 + Y_nmt**2 )
        
        # angulo_ = []
        # Rv_ = np.sqrt( X_s**2 + Y_s**2 )
        Rv_dist_ = []
        Rv_vector_dist_ = []
        NMT_ = []
        for j in range(l):
            for i in range(l):
                if  0 < int(x[j,i]) < 1024 and 0 < int(y[j,i]) < 1024 and int(mascara10[ int(x[j,i]), int(y[j,i]) ]) == 1:
                    Rv_dist_.append( Rv_[j,i]*ps )
                    NMT_.append( res[j,i] )
                    
                    # if Rv_[j,i] != 0:
                    #     angulo = np.arccos( np.dot([X_nmt[j,i], Y_nmt[j,i]], [ x[j,i] - CM[1], y[j,i] - CM[0] ])/( np.linalg.norm([ x[j,i] - CM[1], y[j,i] - CM[0] ])*np.linalg.norm([X_nmt[j,i], Y_nmt[j,i]]) )         )
                    # else:
                    #     angulo = 0
                    # angulo_.append(angulo)

        scale0 = 100
        plt.figure()
        plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
        # plt.quiver(x,y,deformacion[1],-deformacion[0], scale = scale0, pivot='tail')
        # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
        # plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
        
        plt.quiver(x,y,deformacion[1],-deformacion[0], res, cmap = cm_crimson, scale = scale0, pivot='tail')
        # plt.quiver(x,y,X_nmt,-Y_nmt, res, cmap = cm_crimson, scale = scale0, pivot='tail')
        barra_de_escala( 10, sep = 1.5, pixel_size = ps_list[cel], font_size = fs, color = 'k', more_text = 'Vf = ' + str(np.round(vf*ps,1) ) + ' µm' )
        plt.xlim( [0,1023] )
        plt.ylim( [1023,0] )
        plt.show()

        # plt.figure()
        # plt.imshow(res)        
        # plt.show()

        cobertura_celula.append( np.mean(desvios_bin) )
        NMT_list_celula.append( np.mean(NMT_) )
        Rv_dist_celula.append(Rv_dist_)
        # angulo_celula.append(angulo_)
    
    cobertura_todas.append( cobertura_celula )
    NMT_list_todas.append( NMT_list_celula ) 
    Rv_dist_todas.append(Rv_dist_celula)
    # angulo_todas.append(angulo_celula)

#%%

ventanas0 = np.arange( int(v_a/ps_list[0]), int(v_b/ps_list[0]),-1)*ps_list[0]
ventanas1 = np.arange( int(v_a/ps_list[1]), int(v_b/ps_list[1]),-1)*ps_list[1]
ventanas2 = np.arange( int(v_a/ps_list[2]), int(v_b/ps_list[2]),-1)*ps_list[2]
ventanas3 = np.arange( int(v_a/ps_list[3]), int(v_b/ps_list[3]),-1)*ps_list[3]
ventanas = [ventanas0,ventanas1,ventanas2,ventanas3]

promR0 = np.array( [ np.mean(Rv_dist_todas[0][i]) for i in range(len( Rv_dist_todas[0]) ) ]  ) 
promR1 = np.array( [ np.mean(Rv_dist_todas[1][i]) for i in range(len( Rv_dist_todas[1]) ) ]  ) 
promR2 = np.array( [ np.mean(Rv_dist_todas[2][i]) for i in range(len( Rv_dist_todas[2]) ) ]  ) 
promR3 = np.array( [ np.mean(Rv_dist_todas[3][i]) for i in range(len( Rv_dist_todas[3]) ) ]  ) 
promR = [promR0, promR1, promR2, promR3 ] 

stdR0 = np.array( [ np.std(Rv_dist_todas[0][i]) for i in range(len( Rv_dist_todas[0]) ) ]  ) 
stdR1 = np.array( [ np.std(Rv_dist_todas[1][i]) for i in range(len( Rv_dist_todas[1]) ) ]  ) 
stdR2 = np.array( [ np.std(Rv_dist_todas[2][i]) for i in range(len( Rv_dist_todas[2]) ) ]  ) 
stdR3 = np.array( [ np.std(Rv_dist_todas[3][i]) for i in range(len( Rv_dist_todas[3]) ) ]  ) 
stdR = [stdR0, stdR1, stdR2, stdR3 ] 
#%%
fig, ax1 = plt.subplots(figsize=(7, 12))
cel = 0
line1, = ax1.plot( ventanas[cel], promR[cel], 'o' , color=colores[cel], label='NMT')
cel = 1
line1, = ax1.plot( ventanas[cel], promR[cel], 'o' , color=colores[cel], label='NMT')
cel = 2
line1, = ax1.plot( ventanas[cel], promR[cel], 'o' , color=colores[cel], label='NMT')
cel = 3
line1, = ax1.plot( ventanas[cel], promR[cel], 'o' , color=colores[cel], label='NMT')

# line3, = ax1.plot([1],[1],'s', c = 'k', label = 'NMT')
ax1.set_ylabel('Deformación promedio [µm]', color='k')
ax1.set_xlabel('Ventana en la última iteración [µm]', color='k')
ax1.set_yticks( np.arange(0.16,0.56, 0.02), np.round(np.arange(0.16,0.56, 0.02),2) )
# ax1.set_ylim([-0.005,0.105])
ax1.set_xlim([5.2,0.8])
ax1.grid()

#%%


#%%
cel = 0
plt.rcParams['font.size'] = 11

fig, ax1 = plt.subplots(figsize=(7, 4))
cel = 0
line1, = ax1.plot( ventanas[cel], np.array(NMT_list_todas[cel]), 's' , color=colores[cel], label='NMT')
cel = 1
line1, = ax1.plot( ventanas[cel], np.array(NMT_list_todas[cel]), 's' , color=colores[cel], label='NMT')
cel = 2
line1, = ax1.plot( ventanas[cel], np.array(NMT_list_todas[cel]), 's' , color=colores[cel], label='NMT')
cel = 3
line1, = ax1.plot( ventanas[cel], np.array(NMT_list_todas[cel]), 's' , color=colores[cel], label='NMT')
line3, = ax1.plot([1],[1],'s', c = 'k', label = 'NMT')
ax1.set_ylabel('Fracción excluida por el NMT', color='k')
ax1.set_xlabel('Ventana en la última iteración [µm]', color='k')
ax1.set_ylim([-0.005,0.105])
ax1.set_xlim([5.2,0.8])
ax1.grid()

ax2 = ax1.twinx()
cel = 0
line2, = ax2.plot( ventanas[cel], cobertura_todas[cel], '^' , color=colores[cel], label='NMT')
cel = 1
line2, = ax2.plot( ventanas[cel], cobertura_todas[cel], '^' , color=colores[cel], label='NMT')
cel = 2
line2, = ax2.plot( ventanas[cel], cobertura_todas[cel], '^' , color=colores[cel], label='NMT')
cel = 3
line2, = ax2.plot( ventanas[cel], cobertura_todas[cel], '^' , color=colores[cel], label='NMT')
line4, = ax2.plot([1],[2],'^', c = 'k', label = 'Cobertura')

ax2.set_ylabel('Fracción cubierta')
ax2.set_ylim([-0.05,1.05])

# Añadir leyendas y mostrar el gráfico

lines = [line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower right')

    
plt.show()

#%%

plt.rcParams['font.size'] = 12
fig, axs = plt.subplots(4, 1, figsize=(7, 11))
orden = [3,2,1,0]

# Trazar en cada subgráfico
for iterador, ax in enumerate(axs):

    cel = orden[iterador]
    
    line1, = ax.plot( ventanas[cel] , promR[cel], 'o', color = colores[cel], label = "Deformación" )
    # line1 = ax.errorbar( ventanas[cel] , promR[cel], stdR[cel]/30, fmt = 'o', color = colores[cel], label = "Deformación" )
    # line1 = ax.plot(ventanas[cel] , NMT_list_todas[cel], 'o'  )
    ax.set_ylabel('Deformación promedio [µm]')
    
    
    if iterador < 3:
        ax.set_ylim([0.09,0.21])
        ax.set_xticks(np.arange(7),['']*7)
    else:
        # ax.set_ylim([0.35,0.5])
        ax.set_xlabel('Ventana de exploración [µm]', color='k')
    ax.set_xlim([5.3,0.7])
    
    
    ax2 = ax.twinx()
    line2, = ax2.plot( ventanas[cel], cobertura_todas[cel], 'v' , color='k', label='Cobertura')
    line3, = ax2.plot( ventanas[cel], np.array(NMT_list_todas[cel])*10, 's' , color='k', label='NMT x10')
    # line2 = ax2.errorbar(np.arange( ventanas[cel] cobertura_todas[cel], D_cobertura_todas[cel], fmt = 'o' , color='k', label='Cobertura')
    ax2.set_ylabel('Parámetros de control', color='k')
    ax2.set_ylim([-0.19,1.19])
    
    # Añadir leyendas y mostrar el gráfico
    ax.grid()
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='lower right')
    
fig.tight_layout()
    
plt.show()
#%%















cel = 0
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
plt.xlim( [m*5.5+b, m*0.8+b ] )
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


# ventanas0 = np.arange( int(v_a/ps_list[0]), int(v_b/ps_list[0]),-1)*ps_list[0]
ventanas0 = np.arange( int(5/ps_list[3]), int(1/ps_list[3]),-1)*ps_list[3]

promR0 = np.array( [ np.mean(Rv_dist_todas[0][i]) for i in range(len( Rv_dist_todas[0]) ) ]  ) 
stdR0 = np.array( [ np.std(Rv_dist_todas[0][i]) for i in range(len( Rv_dist_todas[0]) ) ]  ) 
# std_ang = np.array( [ np.std( angulo_todas[0][i] ) for i in range(len( angulo_todas[0]) ) ]  ) 
# mean_ang = np.array( [ np.mean( angulo_todas[0][i] ) for i in range(len( angulo_todas[0]) ) ]  ) 


plt.figure( figsize = [6,4] )
# plt.errorbar( ventanas0, normalizar(promR0), stdR0/np.max(promR0)/3 , fmt = 'o', label = 'promedio' )
plt.plot( ventanas0, normalizar(promR0), 'o', label = 'promedio' )
# plt.plot( ventanas0, mean_ang , 'o', label = 'angulo' )
plt.plot( ventanas0, cobertura_todas[0], 'o', label = 'cobertura')
plt.plot( ventanas0, np.array(NMT_list_todas[0])/np.max([1]), 'o', label = 'NMT')
plt.xlabel( "Ventana final [µm]" )
plt.grid()
plt.legend()

plt.xlim([5.2,0.8])

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
ws = 2.5

for iterador in range(4):
    cel = iterador
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
    
    it = 3
    vi = int( int( ws/ps )*2**(it-1) )
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

    print( ZYX[0] )
    
    z_0 = ZYX[0] 
    if cel == 2:
        z_0 += 1
        
    j_p, i_p = -1,-1
    Z = Z_iteration0( stack_post0 = stack_post, img_pre0 = pre, win_shape = int( ws/ps ), exploration = 8, translation_Y = Y_nmt, translation_X = X_nmt, mode = "Smooth3", A = As[cel], z0 = z_0 )

    domX.append(x)
    domY.append(y)
    mapasX.append(X_s)    
    mapasY.append(Y_s)
    axial.append(Z)
    mascaras.append(mascara)

#%%
plt.figure( figsize=[8,6], tight_layout = True )
fs = 'small'
ubicaciones = [3,7,11,4,8,12,2,6,10,1,5,9]
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
    plt.imshow( R, vmin = 0, vmax = np.max(R), extent = [x[0,0]-int( ws/ps )/2, x[-1,-1]+int( ws/ps )/2,x[-1,-1]+int( ws/ps )/2,x[0,0]-int( ws/ps )/2] )
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'w', text = False )

    plt.subplot(3, 4, ubicaciones[3*cel+2] )
    plt.imshow( smooth(Z, 1) , extent = [x[0,0]-int( ws/ps )/2, x[-1,-1]+int( ws/ps )/2,x[-1,-1]+int( ws/ps )/2,x[0,0]-int( ws/ps )/2])
    barra_de_escala( 20, sep = 1.5,  font_size = fs, color = 'w', text = False )    


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
z_0_list = [5,5,4,6]
ccm_list_todas = []

z_f = 3
for iterador in range(4):
    cel = iterador
    runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
    
    ccm_list = []
    z_0 = z_0_list[cel]
    pre = stack_pre[z_0]
    for z_i in range(-z_f, z_f + 1,1):
        post, ccm, YX = correct_driff( stack_post[z_0 + z_i], pre, 20, info = True )
        ccm_list.append(ccm)
    ccm_list_todas.append(ccm_list)

#%%
cel = 3

plt.plot( np.arange(len(ccm_list))-z_f, normalizar(ccm_list_todas[cel]), 'o', c = colores[cel] )
















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




#%%







#%%

A = As[cel]
ws = 2.5
pre = stack_pre[5]
desvios_bin, limit = busca_esferas( pre, ps = ps, th = A, win = ws )
plt.figure( figsize = [6, 6] )
plt.imshow( pre[ :limit , :limit ], cmap = cm_crimson, vmin = 150, vmax = 600 )
plt.imshow( desvios_bin, cmap = 'gray', alpha = 0.09, extent = [0,limit,limit,0])
barra_de_escala( 10, sep = 2, more_text = 'Célula ' + str(cel), a_lot_of_text = str( int( np.mean( desvios_bin )*100 ) ) + '%', font_size = 'large' )
print(  np.mean(desvios_bin) )

#%%
cel = 0
runcell('Invocacion', 'C:/Users/gonza/1/Tesis/PIV/Puesta a punto 2.py')
ws = 2.5
it = 3
vi = int( int( np.round(ws/ps) )*2**(it-1) )
bordes_extra = 8

Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth3"
suave0 = 3

pre = stack_pre[1]
post, ZYX = correct_driff_3D( stack_post, pre, 20, info = True )

control0 = [np.random.randint(32),np.random.randint(32)]
dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo, A = As[cel], control = control0)
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
x, y = dominio
print(  np.mean(res) )

#%%
fs = 'small'

scale0 = 100
plt.figure(figsize = [6,4] )
plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
# plt.quiver(x,y,X_0,-Y_0, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
barra_de_escala( 10, sep = 1.5,  font_size = fs, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.show()


#%%

a, b = 19, 28

w = 31
a2 = np.mean(pre)/np.mean(post)

pre1_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))]*a2 
post0_grande = post[int(w*a-8) : int(w*(a+1)+8), int(w*b-8) : int(w*(b+1)+8)]*a2 


plt.figure()
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)


plt.figure()
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)

#%%
img_pre, big_img_post = pre1_chico, post0_grande
l = img_pre.shape[0]
cross_corr = smooth( signal.correlate(big_img_post - big_img_post.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft"), 3 ) 
y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
cross_corr_max = cross_corr[y0, x0]
y, x = -(y0 - b), -(x0 - b)

#%%

pre_win, post_win, win_shape = pre1_chico, post0_chico, w
plt.figure( tight_layout=True )
plt.subplot(2,2,1)
plt.imshow( pre_win, cmap = cm_crimson, vmin = 100, vmax = 600 )
barra_de_escala( 1, img_len = win_shape, sep = 0.1, pixel_size = ws/win_shape, font_size = 'xx-large', color = 'w' )

plt.subplot(2,2,2)
plt.imshow( post_win, cmap = cm_green, vmin = 100, vmax = 600 )
barra_de_escala( 1, img_len = win_shape, sep = 0.1, pixel_size = ws/win_shape, font_size = 'xx-large', color = 'w' )
 
plt.subplot(2,2,3)
plt.imshow( pre_win, cmap = cm_crimson, vmin = 100, vmax = 400 )
plt.imshow( post_win, cmap = cm_green, vmin = 100, vmax = 400, alpha = 0.6 )
barra_de_escala( 1, img_len = win_shape, sep = 0.1, pixel_size = ws/win_shape, font_size = 'xx-large', color = 'w' )
 
plt.subplot(2,2,4)
l_cc = len(cross_corr)
plt.imshow( np.fliplr(np.flipud(cross_corr)) , cmap = cm_orange )
plt.plot( [1],[5], 'x', c = 'k', ms = 10 )
# plt.plot( [3],[3], 'o', c = marca, ms = 40 )
barra_de_escala( 0.5, img_len = l_cc-0.3, sep = 0.06, pixel_size = ws/win_shape, font_size = 'xx-large', color = 'w' )
# plt.xlim([l_cc-0.5,-0.5])
# plt.ylim([l_cc-0.5,-0.5])

plt.show()













#%%
ws = 10
desvios_bin, limit = busca_esferas( pre, ps = ps, th = A, win = ws )
plt.figure( figsize = [6, 6] )
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 600 )
barra_de_escala( 10, sep = 1, font_size = 'large' )

#%% vi = 124 px

pre_grande = np.ones([1116]*2)*np.max(pre)/8
pre_grande[ 46:-46, 46:-46 ] = pre

#%%


plt.figure( figsize = [8, 6], layout = 'compressed' )

plt.subplot(1,3,1)
plt.imshow( pre_grande, cmap = cm_crimson, vmin = 150, vmax = 600 )
# barra_de_escala( 10, img_len = 1100, sep = 1, font_size = 'large' )
plt.xticks([])
plt.yticks([])
plt.text(10, -10, '10 µm', ha='left', va = 'bottom', fontsize = 14, weight='bold')

for i in np.arange(1,9,1):
    plt.plot( [124*i,124*i], [0,1115], c = 'w', ls = '--' , lw = 0.5)
    plt.plot( [0,1115], [124*i,124*i], c = 'w', ls = '--' , lw = 0.5)

plt.subplot(1,3,2)
plt.imshow( pre_grande, cmap = cm_crimson, vmin = 150, vmax = 600 )
# barra_de_escala( 10, img_len = 1100, sep = 1, font_size = 'large' )
plt.xticks([])
plt.yticks([])
plt.text(10, -10, '5 µm', ha='left', va = 'bottom', fontsize = 14, weight='bold')

for i in np.arange(0.5,9,0.5):
    plt.plot( [124*i,124*i], [0,1115], c = 'w', ls = '--' , lw = 0.25)
    plt.plot( [0,1115], [124*i,124*i], c = 'w', ls = '--' , lw = 0.25)

plt.subplot(1,3,3)
plt.imshow( pre_grande, cmap = cm_crimson, vmin = 150, vmax = 600 )
# barra_de_escala( 10, img_len = 1100, sep = 1, font_size = 'large' )
plt.xticks([])
plt.yticks([])
plt.text(10, -10, '2.5 µm', ha='left', va = 'bottom', fontsize = 14, weight='bold')

for i in np.arange(0.25,9,0.25):
    plt.plot( [124*i,124*i], [0,1115], c = 'w', ls = '--' , lw = 0.125)
    plt.plot( [0,1115], [124*i,124*i], c = 'w', ls = '--' , lw = 0.125)











