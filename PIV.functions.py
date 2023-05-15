# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:14:57 2023

@author: gonza
"""

import numpy as np
import imageio.v3 as iio
from scipy import signal    # Para aplicar filtros
from wand.image import Image

#%%
def centrar_referencia(imagen_post, imagen_pre, bordes_extra, maximo = False):
    l = imagen_pre.shape[0]
    b = bordes_extra
    imagen_post_grande = np.ones([l+2*b, l+2*b])*np.mean( imagen_post.flatten() )
    imagen_post_grande[ b:-b , b:-b ] = imagen_post
    cross_corr = signal.correlate(imagen_post_grande - imagen_post_grande.mean(), imagen_pre - imagen_pre.mean(), mode = 'valid', method="fft")
    y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
    cross_corr_max = cross_corr[y0, x0]
    y, x = -(y0 - bordes_extra), -(x0 - bordes_extra)
    if maximo:
        devolver = imagen_post_grande[ b-y:-b-y , b-x:-b-x ], cross_corr_max
    else:
        devolver = imagen_post_grande[ b-y:-b-y , b-x:-b-x ]
    return devolver

def arma_pares(stack_post, stack_pre, bordes_extra):
    return "Funcion en proceso"


def una_iteracion( imagen_post, imagen_pre, tamano_de_la_ventana, bordes_extra = 10, traslacion_Y = np.zeros([1,1]), traslacion_X = np.zeros([1,1]), bordes_limite = 0):
    '''
    imagen_post(2D-array): imagen de las esferas con el sustrato relajado
    imagen_pre(2D-array): imagen de las esferas con el sustrato deformado
    tamaño_de_la_ventana(int): lado de la ventana de escaneo, que es cuadrada
    bordes_extra(int): indica cuántos píxeles de borde de más se considerarán en la ventana de la imagen post, lo que determina también las posiciones relativas en las que se estudiará la correlación.
    traslación_X/Y(2D-array): para una segunda o tercera iteración, indica la deformación calculada con una ventana mayor
    bordes_limite(int): indica el margen entre la imagen completa y el conjunto de ventanas de escaneo, estás difieren en tamano en general
    '''
    tamano_de_las_imagenes = imagen_pre.shape[0]
    
    # aca debe calcularse en base a la 1ra ventana y despues se duplica, si no pueden sumarse ventanas extra
    if bordes_limite == 0:
        bordes_limite = ( tamano_de_las_imagenes - tamano_de_la_ventana*(tamano_de_las_imagenes//tamano_de_la_ventana) )//2
        if bordes_limite == 0:
            bordes_limite = tamano_de_la_ventana//2
            
    divis = int( (tamano_de_las_imagenes-2*bordes_limite)/tamano_de_la_ventana  )

    if bordes_extra >= bordes_limite:
        bordes_extra = bordes_limite
        
    if traslacion_Y.all() == 0:
        traslacion_Y = np.zeros([divis,divis])
    if traslacion_X.all() == 0:
        traslacion_X = np.zeros([divis,divis])

    Y = np.zeros([divis,divis])
    X = np.zeros([divis,divis])
    
    for j in range(divis):
        for i in range(divis):
            
            Ay_pre = (j)*tamano_de_la_ventana    +  bordes_limite  + int(traslacion_Y[j,i])
            By_pre = (j+1)*tamano_de_la_ventana  +  bordes_limite  + int(traslacion_Y[j,i])
            Ax_pre = (i)*tamano_de_la_ventana    +  bordes_limite  + int(traslacion_X[j,i])
            Bx_pre = (i+1)*tamano_de_la_ventana  +  bordes_limite  + int(traslacion_X[j,i])

            Ay_post = (j)*(tamano_de_la_ventana)   +  bordes_limite - bordes_extra
            By_post = (j+1)*(tamano_de_la_ventana) +  bordes_limite + bordes_extra
            Ax_post = (i)*(tamano_de_la_ventana)   +  bordes_limite - bordes_extra
            Bx_post = (i+1)*(tamano_de_la_ventana) +  bordes_limite + bordes_extra
            
            pre_win = imagen_pre[ Ay_pre : By_pre, Ax_pre : Bx_pre ]
            post_bigwin = imagen_post[ Ay_post : By_post, Ax_post : Bx_post ]
    
            cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method="fft")
            y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
            y, x = -(y0 - bordes_extra), -(x0 - bordes_extra)
        
            Y[j,i] = y
            X[j,i] = x    

    return Y+traslacion_Y, X+traslacion_X

def n_iteraciones( imagen_post, imagen_pre, ventana_inicial, cantidad_de_iteraciones = 3, bordes_extra = 1000):
    n = cantidad_de_iteraciones   
    
    tamano_de_las_imagenes = imagen_pre.shape[0]
    
    limite = ( tamano_de_las_imagenes - ventana_inicial*(tamano_de_las_imagenes//ventana_inicial) )//2
    if limite == 0:
        limite = ventana_inicial//2    
    
    limite = int( (imagen_post.shape[0] - ventana_inicial*(imagen_post.shape[0]//ventana_inicial) )//2 )
    if limite == 0:
        limite = ventana_inicial//2

    tam0 = ( imagen_post.shape[0]//ventana_inicial )    
    X = np.zeros([tam0, tam0])
    Y = np.zeros([tam0, tam0])

    for n0 in range(n):
        ventana =  ventana_inicial//(2**n0)
        print( n0, ventana )
        Y, X = una_iteracion( imagen_post, imagen_pre, ventana, bordes_extra, duplicar_tamano(Y), duplicar_tamano(X), limite )

    return Y, X

def nmt(Y_, X_, noise, threshold, modo = "promedio"):
    Y = Y_.copy()
    X = X_.copy()
    l = X.shape[0]
    result = np.zeros( [l]*2 )
    means_X = np.zeros( [l]*2 )
    means_Y = np.zeros( [l]*2 )
    for j in range(1, l-1):
        for i in range(1, l-1):
            # valores en la pocision a analizar
            value_X = X[j,i]
            value_Y = Y[j,i]
            # valores de los vecinos
            neighbours_X = np.array( [ X[j+1,i+1], X[j+1,i], X[j+1,i-1], X[j,i-1], X[j-1,i-1], X[j-1,i], X[j-1,i+1], X[j,i+1] ] )
            neighbours_Y = np.array( [ Y[j+1,i+1], Y[j+1,i], Y[j+1,i-1], Y[j,i-1], Y[j-1,i-1], Y[j-1,i], Y[j-1,i+1], Y[j,i+1] ] )
            # medias de los vecinos
            median_X = np.median( neighbours_X )        
            median_Y = np.median( neighbours_Y )
            # residuos
            residual_values_X = ( np.abs(neighbours_X - median_X) )
            residual_values_Y = ( np.abs(neighbours_Y - median_Y) )
            # media de los residuos
            res_median_X = np.median( neighbours_X )        
            res_median_Y = np.median( neighbours_Y )
            if res_median_X == 0:
                res_median_X += noise
            if res_median_Y == 0:
                res_median_Y += noise    
            
            residual_value_X0 = np.abs( ( value_X - median_X )/res_median_X )
            residual_value_Y0 = np.abs( ( value_Y - median_Y )/res_median_Y )
            if residual_value_X0 >= threshold or residual_value_Y0 >= threshold:
                means_X[j,i] = np.mean( neighbours_X ) 
                means_Y[j,i] = np.mean( neighbours_Y ) 
                result[j,i] = 1

    if modo == "promedio":
        # lo cambio por el promedio, despues tengo que averiguar/preguntar bien
        for j in range(1, l-1):
            for i in range(1, l-1):
                if result[j,i] == 1:
                    X[j,i] = means_X[j,i]
                    Y[j,i] = means_Y[j,i]
        X[:,0], X[:,-1], X[-1,:], X[0,:]  = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)
        Y[:,0], Y[:,-1], Y[-1,:], Y[0,:]  = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)

    return Y, X, result

def deformar( imagen_post, grado, tamano, cantidad):
    imagen = np.copy( imagen_post )
    l = imagen.shape[0]
    a = l//( cantidad + 1 )
    d = tamano//2
    for i in range(cantidad):
        for j in range(cantidad):
            cen = [(i+1)*a, (j+1)*a]
            pedazo = imagen[ int(cen[0] - d) : int(cen[0] + d) , int(cen[1] - d) : int(cen[1] + d) ]
            iio.imwrite( "pedazo.tiff", pedazo )
            with Image( filename = "pedazo.tiff" ) as img:
                # Implode
                img.implode(grado)
                img.save( filename = "imp.tiff" )

            implosion = iio.imread('imp.tiff')
            # if cen == [a,a]:
            #     plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.imshow(pedazo, cmap = 'gray', vmin = 80, vmax = 800)
            #     plt.title("pedazo")
            #     plt.subplot(1,2,2)
            #     plt.imshow(implosion, cmap = 'gray', vmin = 80, vmax = 800)
            #     plt.title("implosion")
            #     plt.show()
            
            imagen[ int(cen[0] - d) : int(cen[0] + d) , int(cen[1] - d) : int(cen[1] + d) ] = implosion

    return imagen

def duplicar_tamano(array):
    largo = array.shape[0]*2
    array_grande = np.zeros([largo, largo])
    for j0 in range(largo):
        for i0 in range(largo):
            array_grande[j0,i0] = array[j0//2,i0//2]
    return array_grande

def suavizar(imagen, lado_del_nucleo):
    nucleo = np.ones([lado_del_nucleo]*2)/lado_del_nucleo**2
    return signal.convolve2d(imagen, nucleo, mode='same')
