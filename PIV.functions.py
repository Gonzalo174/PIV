# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:14:57 2023

@author: gonza
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

def four_core(array2D):
    """
    Parameters
    ----------
    array2D : numpy.2darray
        Matrix of the previous iteration displacement in one axis.

    Returns
    -------
    big_array2D : numpy.2darray
        Sice doubeled displacement matrix that have the same shape as the next iteration matrix.

    """
    big_array2D = array2D
    if type( array2D ) == np.ndarray:
        l = len(array2D)*2
        big_array2D = np.zeros([l, l])
        for j0 in range(l):
            for i0 in range(l):
                big_array2D[j0,i0] = array2D[j0//2,i0//2]
    return big_array2D


def round_pro(array2D):
    """
    Parameters
    ----------
    array2D : numpy.2darray
        2 dimentional array.

    Returns
    -------
    round_array2D : numpy.2darray
        The same 2 dimentional array but each element is an intiger.

    """
    round_array2D = array2D
    if type( array2D ) == np.ndarray:
        round_array2D = np.round( array2D )
    return round_array2D


def median_blur(img, kernel_size):
    """
    Parameters
    ----------
    img : numpy.2darray like
        2 dimentional array - image.
    kernel_size : int
        lenght of the kernel used to blur the image
    
    Returns
    -------
    blur : numpy.2darray like
        2 dimentional array - image, each pixel value is the median of the pixels values at kernel´s area, cetered in that pixel.

    """
    L, k = len(img), kernel_size
    img0 = np.ones([L + k//2, L + k//2])*np.mean( img.flatten() )
    img0[k//2:L + k//2, k//2:L + k//2] = img
    blur = np.zeros([L,L])
    for j in range(L):
        for i in range(L):
            muestra = img0[ j: j+k , i: i+k ].flatten()
            media = np.median(muestra)
            blur[j,i] = media
    return blur 


def smooth(img, kernel_size):
    """
    Parameters
    ----------
    img : numpy.2darray like
        2 dimentional array - image.
    kernel_size : int
        lenght of the kernel used to blur the image

    Returns
    -------
    smooth_img : numpy.2darray
        2 dimentional array - image, each pixel value is the mean of the pixels values at kernel´s area, cetered in that pixel.

    """
    kernel = np.ones([kernel_size]*2)/kernel_size**2
    smooth_img = signal.convolve2d(img, kernel, mode='same')
    return smooth_img


def area_upper(binary_map, kernel_size = 50, threshold = 0.1):
    """
    Parameters
    ----------
    binary_map : numpy.2darray like
        2 dimentional array - binary image pixels take 1 value inside cell area and 0 outside.
    kernel_size : int, optional
        lenght of the kernel used to blur the image. The default is 50.
    threshold : float, optional
        Value between 0 and 1, if it is more than 0.5 area increase, if it is less, area decrease. The default is 0.1.

    Returns
    -------
    mask : numpy.2darray like
        Smoothed and rebinarized the cell mask. Now pixels take 1 value inside cell and a border arround it.

    """
    l = len(binary_map)
    mask = np.zeros([l,l])
    smooth_map = smooth( binary_map, kernel_size)
    mask[ smooth_map > threshold ] = 1
    return mask


def contour(img):
    """
    Parameters
    ----------
    img : numpy.2darray like
        2 dimentional array - image.

    Returns
    -------
    bordes : numpy.2darray
        Edges of the input.

    """
    s = [[ 1,  2,  1],  
         [ 0,  0,  0], 
         [-1, -2, -1]]

    HR = signal.convolve2d(img, s)
    VR = signal.convolve2d(img, np.transpose(s))
    bordes = (HR**2 + VR**2)**0.5
    return bordes


def center_of_mass(binary_map):
    """
    Parameters
    ----------
    binary_map : numpy.2darray like
        2 dimentional array - binary image pixels take 1 value inside cell area and 0 outside.

    Returns
    -------
    CM : tuple
        Coordinates of the center of mass of the cell mask.

    """
    CMy = 0
    CMx = 0
    mass = np.sum(binary_map)
    l = len(binary_map)
    for j in range(l):
        for i in range(l):
            CMy += j*binary_map[j,i]/mass
            CMx += i*binary_map[j,i]/mass
    CM = ( int(CMy), int(CMx) )
    return CM 


def correct_driff(img_post, img_pre, exploration, info = False):
    """
    Parameters
    ----------
    img_post : numpy.2darray like
        2 dimentional array - image of the nanospheres after removing the cells.
    img_pre : numpy.2darray like
        2 dimentional array - image of the nanospheres with the cells adhered on the hydrogel.
    exploration : int
        Number of pixels explored over the plane
    info : bool, optional
        If it is true also returns the value of the maximum correlation and its position. The default is False.

    Returns
    -------
    devolver : numpy.2darray
        post image driff corrected.

    """
    l = img_pre.shape[0]
    b = exploration
    big_img_post = np.ones([l+2*b, l+2*b])*np.mean( img_post.flatten() )
    big_img_post[ b:-b , b:-b ] = img_post
    cross_corr = smooth( signal.correlate(big_img_post - big_img_post.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft"), 3 ) 
    y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
    cross_corr_max = cross_corr[y0, x0]
    y, x = -(y0 - b), -(x0 - b)
    if info:
        devolver = big_img_post[ b-y:-b-y , b-x:-b-x ], cross_corr_max, (y,x)
    else:
        devolver = big_img_post[ b-y:-b-y , b-x:-b-x ]
    return devolver


def correct_driff_3D(stack_post, img_pre, exploration, info = False):
    """
    Parameters
    ----------
    stack_post : numpy.3darray like
        2 dimentional array - stack of the nanospheres after removing the cells.
    img_pre : numpy.2darray like
        2 dimentional array - image of the nanospheres with the cells adhered on the hydrogel.
    exploration : int
        Number of pixels explored over the plane
    info : bool, optional
        If it is true also returns the position of the maximum correlation. The default is False.

    Returns
    -------
    devolver : numpy.2darray
        post image driff corrected.

    """
    images_post = []
    corr = []
    XY = []
    for z in range(len(stack_post)):
        img_post = stack_post[z]
        img_post_centered, cross_corr_max, (y,x) = correct_driff(img_post, img_pre, exploration, info = True)
        images_post.append( img_post_centered )
        corr.append( cross_corr_max )
        XY.append( (y,x) )
        
    
    if info:
        devolver = images_post[ np.argmax(corr) ], ( np.argmax(corr), XY[np.argmax(corr)][0], XY[np.argmax(corr)][1] )
    else:
        devolver = images_post[ np.argmax(corr) ]
    
    return devolver


def curve_fit_pro(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, method=None, jac=None, full_output=False):
    try:
        # Attempt to perform the curve fit
        popt = p0
        popt, pcov = curve_fit(f, xdata, ydata, p0, maxfev = 10000)#, sigma, absolute_sigma, check_finite, method, jac, full_output )

    except RuntimeError:
        popt, pcov = p0, np.ones([len(p0),len(p0)])
        
    return popt, pcov



def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))) + offset
    return g.ravel()

def gaussian_2d_plot(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))) + offset
    return g


def iteration( img_post, img_pre, win_shape, exploration = 10, translation_Y = "None", translation_X = "None", max_exploration = 0, mode = "Default"):
    """
    Parameters
    ----------
    img_post : numpy.2darray like
        2 dimentional array - image of the nanospheres after removing the cells.
    img_pre : numpy.2darray like
        2 dimentional array - image of the nanospheres with the cells adhered on the hydrogel.
    win_shape : int
        Exploration windows side lenght in pixels.
    exploration : int, optional
        Number of pixels explored over the plane for each exploration window. The default is 10.
    translation_Y : numpy.2darray like, optional
        Deformation map obtenied in a previous iteration using a windows of twice side leght. The default is "None".
    translation_X : numpy.2darray like, optional
        Deformation map obtenied in a previous iteration using a windows of twice side leght. The default is "None".
    max_exploration : int, optional
        Maximum exploration. The default is 0.
    mode : str, optional
        Mode using to calculate maximum correlation. The default is "Default".

    Returns
    -------
    deformation_map : 2 numpy.2darray
        Arrays containing the resulting deformation in Y and X, that is the sum of the previous deformation maps and the calculated position of the cross correlation maximums. 

    """
    img_shape = img_pre.shape[0]
    
    # aca debe calcularse en base a la 1ra ventana y despues se duplica, si no pueden sumarse ventanas extra
    if max_exploration == 0:
        max_exploration = ( img_shape - win_shape*(img_shape//win_shape) )//2
        if max_exploration == 0:
            max_exploration = win_shape//2
            
    divis = int( (img_shape - 2*max_exploration)/win_shape  )

    Y = np.zeros([divis,divis])
    X = np.zeros([divis,divis])

    if exploration >= max_exploration:
        exploration = max_exploration
    
    if type( translation_Y ) == str:
        translation_Y = np.zeros([divis,divis])
    if type( translation_X ) == str:
        translation_X = np.zeros([divis,divis])
        
    cc_area = signal.correlate(img_post - img_post.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft")/(img_shape**2)
    factor = 0.3

    for j in range(divis):
        for i in range(divis):

            Ay_pre = (j)*win_shape    +  max_exploration  + int(translation_Y[j,i])
            By_pre = (j+1)*win_shape  +  max_exploration  + int(translation_Y[j,i])
            Ax_pre = (i)*win_shape    +  max_exploration  + int(translation_X[j,i])
            Bx_pre = (i+1)*win_shape  +  max_exploration  + int(translation_X[j,i])

            Ay_post = (j)*(win_shape)   +  max_exploration - exploration
            By_post = (j+1)*(win_shape) +  max_exploration + exploration
            Ax_post = (i)*(win_shape)   +  max_exploration - exploration
            Bx_post = (i+1)*(win_shape) +  max_exploration + exploration
            
            pre_win = img_pre[ Ay_pre : By_pre, Ax_pre : Bx_pre ]
            post_bigwin = img_post[ Ay_post : By_post, Ax_post : Bx_post ]
    
            cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method = "fft") 
            
            if mode[:-1] == "Smooth" or mode == "Smooth":
                ks = 3
                if mode[-1] != "h":
                    ks = int(mode[-1])
                cross_corr = smooth( cross_corr , ks )
                
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                
                cc_area_iteration = cross_corr[y0,x0]/(win_shape**2)
                if cc_area_iteration > cc_area*factor:
                    Y[j,i] = y
                    X[j,i] = x
                    
            if mode == "Fit":
                cross_corr = smooth( cross_corr , 3 )
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                yo, xo = -y, -x
                
                cc_area_iteration = cross_corr[y0,x0]/(win_shape**2)
                if cc_area_iteration > cc_area*factor:
                    data = cross_corr
                    u, v = np.meshgrid(np.linspace(-exploration, exploration, 2*exploration+1), np.linspace(-exploration, exploration, 2*exploration+1) )
                    amplitude0 = np.max(data)-np.min(data)
                    
                    popt = [amplitude0, xo, yo, 3, 3, 0, np.min(data)]
                    popt, pcov = curve_fit_pro(gaussian_2d, (u, v), data.ravel(), p0 = popt )
                    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
                    
                    Y[j,i] = -yo
                    X[j,i] = -xo
                    
                    if win_shape == 32 and i%4==1 and j%4==0 and mapas:
                        plt.figure()
                        plt.yticks( np.arange( exploration )*2 +1, -(np.arange( exploration )*2 + 1 - exploration) )
                        plt.xticks( np.arange( exploration )*2 +1, np.arange( exploration )*2 + 1 - exploration )
                        plt.xlabel("Distancia [px]")
                        plt.ylabel("Distancia [px]")
                        plt.title("Correlación cruzada: "+str(i)+"-"+str(j))
                        plt.imshow( np.flip(smooth( cross_corr , 3 ),1), vmax = 1000000 )
                        plt.plot( [exploration+x], [y0], 'o',c = 'red', markersize = 10 )
                        plt.plot( [exploration-xo], [exploration+yo], 'o',c = 'green', markersize = 10 )

            if mode == "No control":
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                
                Y[j,i] = y
                X[j,i] = x
            
            if mode == "Default":
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                
                cc_area_iteration = cross_corr[y0,x0]/(win_shape**2)
                if cc_area_iteration > cc_area*factor:
                    Y[j,i] = y
                    X[j,i] = x  
                    
    deformation_map = Y+translation_Y, X+translation_X       
            
    return deformation_map


def n_iterations( img_post, img_pre, win_shape_0, iterations = 3, exploration = 1000, mode = "Default"):
    """
    Parameters
    ----------
    img_post : numpy.2darray like
        2 dimentional array - image of the nanospheres after removing the cells.
    img_pre : numpy.2darray like
        2 dimentional array - image of the nanospheres with the cells adhered on the hydrogel.
    win_shape_0 : int
        Exploration windows side lenght in pixels for the first iteration.
    iterations : int, optional
        Number of iterarions to do. The default is 3.
    exploration : int, optional
        Number of pixels explored over the plane for each exploration window. The default is 1000.
    mode : str, optional
        Mode using to calculate maximum correlation in the last iteration. The default is "Default".

    Returns
    -------
    deformation_map : 2 numpy.2darray
        Arrays containing the resulting deformation in Y and X, that is the sum of the deformation calculated using the position of the cross correlation maximum at the iterations. 


    """
    n = iterations   
    
    img_shape = img_pre.shape[0]
    
    limite = ( img_shape - win_shape_0*(img_shape//win_shape_0) )//2
    if limite == 0:
        limite = win_shape_0//2    
    
    limite = int( (img_post.shape[0] - win_shape_0*(img_post.shape[0]//win_shape_0) )//2 )
    if limite == 0:
        limite = win_shape_0//2

    X = "None" #np.zeros([tam0//2, tam0//2])
    Y = "None" #np.zeros([tam0//2, tam0//2])

    mode_array = ["Smooth3"]*(n-1) + [mode]
    for n0 in range(n):
        ventana =  win_shape_0//(2**n0)
        print( n0, ventana )
        Y, X = iteration( img_post, img_pre, ventana, exploration, four_core(Y), four_core(X), limite, mode_array[n0] )

    deformation_map = Y, X          

    return deformation_map

def nmt(Y_, X_, noise, threshold, mode = "Mean"):
    """
    Parameters
    ----------
    Y_ : numpy.2darray
        Deformation map coortinate.
    X_ : numpy.2darray
        Deformation map coortinate.
    noise : Float
        Noise added to prevent cero division.
    threshold : Float
        Threshold to detect erroneous deformation values.
    mode : str, optional
        Metodh use to replace . The default is "Mean".

    Returns
    -------
    Y : numpy.2darray
        Deformation map coortinate.
    X : numpy.2darray
        Deformation map coortinate.
    result : numpy.2darray
        DESCRIPTION.

    """
    Y = Y_.copy()
    X = X_.copy()
    l = X.shape[0]
    result = np.zeros( [l]*2 )
    # means_X = np.zeros( [l]*2 )
    # means_Y = np.zeros( [l]*2 )
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
            res_median_X = np.median( residual_values_X )        
            res_median_Y = np.median( residual_values_Y )
            if res_median_X == 0:
                res_median_X += noise
            if res_median_Y == 0:
                res_median_Y += noise    
            
            residual_value_X0 = np.abs( ( value_X - median_X )/res_median_X )
            residual_value_Y0 = np.abs( ( value_Y - median_Y )/res_median_Y )
            if residual_value_X0 >= threshold or residual_value_Y0 >= threshold:
                # means_X[j,i] = np.mean( neighbours_X ) 
                # means_Y[j,i] = np.mean( neighbours_Y ) 
                result[j,i] = 1

    if mode == "Mean":
        # lo cambio por el promedio, despues tengo que averiguar/preguntar bien
        for j in range(1, l-1):
            for i in range(1, l-1):
                if result[j,i] == 1:
                    
                    neighbours_X0 = X[j-1:j+2 , i-1:i+2].flatten()
                    neighbours_Y0 = Y[j-1:j+2 , i-1:i+2].flatten()
                    valid = 1 - result[j-1:j+2 , i-1:i+2].flatten()
                    
                    if sum(valid) != 0:
                        X[j,i] = sum( neighbours_X0*valid )/sum(valid) 
                        Y[j,i] = sum( neighbours_Y0*valid )/sum(valid) 

                    # X[j,i] = means_X[j,i]
                    # Y[j,i] = means_Y[j,i]
                    
        # X[:,0], X[:,-1], X[-1,:], X[0,:]  = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)
        # Y[:,0], Y[:,-1], Y[-1,:], Y[0,:]  = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l)
        X[:,0], X[:,-1], X[-1,:], X[0,:]  = X[:,0]*(1-result[:,0]), X[:,-1]*(1-result[:,-1]), X[-1,:]*(1-result[-1,:]), X[0,:]*(1-result[0,:])
        Y[:,0], Y[:,-1], Y[-1,:], Y[0,:]  = Y[:,0]*(1-result[:,0]), Y[:,-1]*(1-result[:,-1]), Y[-1,:]*(1-result[-1,:]), Y[0,:]*(1-result[0,:])

    return Y, X, result





#%%
from wand.image import Image
import imageio.v3 as iio


def deformar( img_post, grado, tamano, cantidad):
    img = np.copy( img_post )
    l = img.shape[0]
    a = l//( cantidad + 1 )
    d = tamano//2
    for i in range(cantidad):
        for j in range(cantidad):
            cen = [(i+1)*a, (j+1)*a]
            pedazo = img[ int(cen[0] - d) : int(cen[0] + d) , int(cen[1] - d) : int(cen[1] + d) ]
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
            
            img[ int(cen[0] - d) : int(cen[0] + d) , int(cen[1] - d) : int(cen[1] + d) ] = implosion

    return img










