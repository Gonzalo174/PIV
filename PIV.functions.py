# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:14:57 2023

@author: gonza
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

def busca_esferas( img, ps = 0.1007, win = 3, th = 1, std_img = 0 ):
    if std_img == 0:
        std_img = np.std( img )
    
    ws = int( np.round( win/ps ) )
    l = int( len(img)/ws )
    std_matrix = np.zeros( [l]*2 )
    
    for j in range(l):
        for i in range(l):
            img_ = img[ int(ws*j) : int(ws*(j+1)), int(ws*i) : int(ws*(i+1)) ]
            std_matrix[j,i] = np.std( img_ )
    
    win_with_sph = np.zeros( [l]*2 )
    win_with_sph[ std_matrix/std_img > th ] = 1
    
    return win_with_sph, int(ws*l)

def busca_manchas(img, th = 1000):
    sat = np.zeros([1024]*2)
    sat[ pre > th ] = 1
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    return sat

def barra_de_escala( scale_length, pixel_size = 0.1007, scale_unit = 'µm', loc = 'lower right', sep = 1, img_len = 1000, font_size = "x-large", color = 'white', text = True, more_text = '', a_lot_of_text = '' ):
    scale_pixels = scale_length/pixel_size
    scale_bar_length = int( scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    

    for i in np.arange(0, -sep/pixel_size, -0.1):
        plt.plot([ img_len - sep/pixel_size - scale_pixels, img_len - sep/pixel_size ], [img_len - sep/pixel_size + i, img_len - sep/pixel_size + i], color=color, linewidth = 2)
    if text:
        plt.text(img_len - sep/pixel_size - scale_pixels/2, img_len - 3*sep/pixel_size , f'{scale_length} {scale_unit}', color=color, weight='bold', ha='center', va = 'bottom', fontsize = font_size )

    if more_text != '':    
        plt.text(img_len - sep/pixel_size/2, sep/pixel_size , more_text, color=color, weight='bold', ha='right', va = 'top', fontsize = font_size )
    if a_lot_of_text != '':
        plt.text(sep/pixel_size, img_len - sep/pixel_size/2 , a_lot_of_text, color=color, weight='bold', ha='left', va = 'bottom', fontsize = font_size )

    plt.xticks([])
    plt.yticks([])

def barra_de_escala0( scale_length, pixel_size = 0.1007, scale_unit = 'µm', loc = 'lower right', sep = 1, img_len = 1000, font_size = "x-large", color = 'white', text = True, more_text = '', a_lot_of_text = '', delta = 1 ):
    scale_pixels = scale_length/pixel_size
    scale_bar_length = int( scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    
    if loc == 'lower right':
        start_x = int( img_len - ( scale_length + sep )/ps )
        start_y = int( img_len - ( sep )/ps - 1 )
        
    for i in np.arange(0, sep/pixel_size, 0.1):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i, start_y + i], color=color, linewidth = 1)
    if text:
        plt.text(start_x + scale_pixels/2, start_y - delta*sep/pixel_size , f'{scale_length} {scale_unit}', color=color, weight='bold', ha='center', fontsize = font_size )
    if more_text != '':    
        plt.text(img_len - sep/pixel_size/2, sep/pixel_size , more_text, color=color, weight='bold', ha='right', va = 'top', fontsize = font_size )
    if a_lot_of_text != '':
        plt.text(sep/pixel_size, img_len - sep/pixel_size/2 , a_lot_of_text, color=color, weight='bold', ha='left', va = 'bottom', fontsize = font_size )

    plt.xticks([])
    plt.yticks([])


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

def normalizar(array):
    array = np.array(array)
    return (array - min(array))/max(array - min(array))


def desvio_por_plano(stack):
    desvios = np.zeros(len(stack))
    for z in range(len(stack)):
        desvios[z] = np.std( stack[z] )
    return normalizar( desvios )

def border(img, y0, k = 3):
    vecinos = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    mascara = np.zeros(img.shape)
    img_s = smooth( img, k )
    mascara[ img_s > 0.5 ] = 1

    y_borde = [ y0 ]
    x_borde = [    ]
    j = 0
    while len(x_borde) == 0:
        if mascara[600,j] == 1:
            x_borde.append(j-1)
        j += 1    

    seguir = True
    while seguir:
        x0 = x_borde[-1] 
        y0 = y_borde[-1]
        for j in range(8):
            v0 = mascara[ y0 + vecinos[j-1][0], x0 + vecinos[j-1][1] ]
            v1 = mascara[   y0 + vecinos[j][0],   x0 + vecinos[j][1] ]
            if v0 == 0 and v1 == 1:
                x_borde.append( x0 + vecinos[j-1][1] )
                y_borde.append( y0 + vecinos[j-1][0] )
        if ( x_borde[-1] == x_borde[0] and y_borde[-1] == y_borde[0] and len(x_borde) > 1 ) or len(x_borde) > 10000:
            seguir = False

    borde = np.concatenate( (  np.reshape( np.array( y_borde ), [1, len(y_borde)]) , np.reshape( np.array( x_borde ) ,  [1, len(y_borde)] ) ) , axis = 0 )

    return borde



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


def unrotate(img_post, img_pre, exploration_px, exploration_angle = 1, N_angle = 100, info = False):
    """
    Parameters
    ----------
    img_post : numpy.2darray like
        2 dimentional array - image of the nanospheres after removing the cells.
    img_pre : numpy.2darray like
        2 dimentional array - image of the nanospheres with the cells adhered on the hydrogel.
    exploration_px : int
        Number of pixels explored over the plane
    exploration_angle : float
        Set the limits for angle exploration     
    N_angle : int
        Number of angles explored  
    info : bool, optional
        If it is true also returns the value of the maximum correlation and its coordinates. The default is False.

    Returns
    -------
    devolver : numpy.2darray
        post image driff and rotation corrected.

    """
    l = img_pre.shape[0]
    b = exploration_px
    angles = np.linspace(-exploration_angle, exploration_angle, N_angle)
    cc = []
    for ang in angles:
        big_img_post = np.ones([l+2*b, l+2*b])*np.mean( img_post.flatten() )
        big_img_post[ b:-b , b:-b ] = img_post
        # print( len(big_img_post) )
        img_post_rot = ndimage.rotate( big_img_post, ang, reshape = False )
        # print( len(img_post_rot) )
        cross_corr = smooth( signal.correlate(img_post_rot - img_post_rot.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft"), 3 ) 
        cc.append( np.max(cross_corr) )

    img_post_rot = ndimage.rotate(big_img_post, angles[np.array(cc).argmax()], reshape = False  )
    cross_corr = smooth( signal.correlate(img_post_rot - img_post_rot.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft"), 3 ) 
    y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
        
    y, x = -(y0 - b), -(x0 - b)
    
    if info:
        devolver = img_post_rot[ b-y:-b-y , b-x:-b-x ], np.array(cc).argmax(), (y,x,angles[np.array(cc).argmax()]) 
    else:
        devolver = img_post_rot[ b-y:-b-y , b-x:-b-x ]
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
    YX = []
    for z in range(len(stack_post)):
        img_post = stack_post[z]
        img_post_centered, cross_corr_max, (y,x) = correct_driff(img_post, img_pre, exploration, info = True)
        images_post.append( img_post_centered )
        corr.append( cross_corr_max )
        YX.append( (y,x) )
        
    
    if info:
        devolver = images_post[ np.argmax(corr) ], ( np.argmax(corr), YX[np.argmax(corr)][0], YX[np.argmax(corr)][1] )
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
   


def iteration( img_post0, img_pre0, win_shape, exploration = 10, translation_Y = "None", translation_X = "None", edge = 100, mode = "Default", A = 0.9, control = [(-1,-1)]):
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
    edge : int, optional
        Extra border, to prevent going outside post image. The default is 0.
    mode : str, optional
        Mode using to calculate maximum correlation. The default is "Default".
    A : float, optional:
        Controls the admisibility of PRE windows in relation to the presence of signal from nanospheres.

    Returns
    -------
    deformation_map : 2 numpy.2darray
        Arrays containing the resulting deformation in Y and X, that is the sum of the previous deformation maps and the calculated position of the cross correlation maximums. 

    """
    img_shape = img_pre0.shape[0]
    img_pre, img_post = np.ones([img_shape+2*edge,img_shape+2*edge])*np.mean(img_pre0), np.ones([img_shape+2*edge,img_shape+2*edge])*np.mean(img_post0)
    img_pre[edge:-edge,edge:-edge], img_post[edge:-edge,edge:-edge] = img_pre0, img_post0    

    remain = win_shape - img_shape%win_shape
    if remain == win_shape:
        remain = 0
    
    if type( translation_Y ) != str:
        divis = len(translation_Y)
        edge -= (divis*win_shape - img_shape)//2
    elif img_shape%win_shape != 0:  # aca se mete la 1ra si la imagen no encaja con el tamano de ventana
        divis = int( ( img_shape + remain )/win_shape )
        edge -= remain//2 
    else:  # aca se mete la primera si encaja con el tamano de ventana
        divis = int( ( img_shape + remain )/win_shape )
        
    Y = np.zeros([divis,divis])
    X = np.zeros([divis,divis])
    
    if type( translation_Y ) == str:
        translation_Y = np.zeros([divis,divis])
    if type( translation_X ) == str:
        translation_X = np.zeros([divis,divis])
        
    pre_std = np.std(img_pre0)
    post_std = np.std(img_post0)

    for j in range(1,divis-1,1):
        for i in range(1,divis-1,1):

            Ay_pre = (j)*win_shape    +  edge  + int(translation_Y[j,i])
            By_pre = (j+1)*win_shape  +  edge  + int(translation_Y[j,i])
            Ax_pre = (i)*win_shape    +  edge  + int(translation_X[j,i])
            Bx_pre = (i+1)*win_shape  +  edge  + int(translation_X[j,i])

            Ay_post = (j)*(win_shape)   +  edge - exploration
            By_post = (j+1)*(win_shape) +  edge + exploration
            Ax_post = (i)*(win_shape)   +  edge - exploration
            Bx_post = (i+1)*(win_shape) +  edge + exploration
            
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
                
                pre_win_std = np.std( pre_win )
                post_win_std = np.std( post_bigwin[exploration:-exploration, exploration:-exploration] ) 
                marca = 'r'
                if pre_win_std > A*pre_std and post_win_std > A*post_std:
                    Y[j,i] = y
                    X[j,i] = x
                    marca = 'g'

                punto = (j,i) 
                if punto in control:
                # R = np.sqrt( y**2 + x**2 )
                # if R > 2 and (0,0) in control:
                    plt.figure( tight_layout=True )
                    plt.subplot(2,2,1)
                    plt.imshow( pre_win, cmap = cm_crimson, vmin = 100, vmax = 600 )
                    barra_de_escala( 1, img_len = win_shape, sep = 0.15, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )

                    plt.subplot(2,2,2)
                    plt.imshow( post_bigwin[exploration:-exploration,exploration:-exploration], cmap = cm_green, vmin = 100, vmax = 600 )
                    barra_de_escala( 1, img_len = win_shape, sep = 0.15, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )
                 
                    plt.subplot(2,2,3)
                    plt.imshow( pre_win, cmap = cm_crimson, vmin = 100, vmax = 600 )
                    plt.imshow( post_bigwin[ exploration - y : -exploration - y , exploration - x : -exploration - x], cmap = cm_green, vmin = 100, vmax = 600, alpha = 0.6 )
                    barra_de_escala( 1, img_len = win_shape, sep = 0.15, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )
             
                    plt.subplot(2,2,4)
                    l_cc = len(cross_corr)
                    plt.imshow( np.fliplr(np.flipud(cross_corr)) , cmap = 'Oranges' )
                    plt.plot( [x + exploration],[y + exploration], 'o', c = 'b', ms = 10 )
                    plt.plot( [3],[3], 'o', c = marca, ms = 40 )
                    barra_de_escala( 0.5, img_len = l_cc, sep = 0.08, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )
                    # plt.xlim([l_cc-0.5,-0.5])
                    # plt.ylim([l_cc-0.5,-0.5])

                    plt.show()

        
            if mode == "Fit":
                cross_corr = smooth( cross_corr , 3 )
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                yo, xo = -y, -x
                
                pre_win_std = np.std( pre_win )
                marca = 'r'
                if pre_win_std > A*pre_std:
                    data = cross_corr
                    u, v = np.meshgrid(np.linspace(-exploration, exploration, 2*exploration+1), np.linspace(-exploration, exploration, 2*exploration+1) )
                    amplitude0 = np.max(data)-np.min(data)
                    
                    popt = [amplitude0, xo, yo, 3, 3, 0, np.min(data)]
                    popt, pcov = curve_fit_pro(gaussian_2d, (u, v), data.ravel(), p0 = popt )
                    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
                    
                    Y[j,i] = -yo
                    X[j,i] = -xo
                    marca = 'g'
                    
                punto = (j,i) 
                if punto in control:
                # if pre_win_std < A*pre_std:   
                    plt.figure( tight_layout=True )
                    plt.subplot(2,2,1)
                    plt.imshow( pre_win, cmap = 'Reds', vmin = 100, vmax = 600 )
                    barra_de_escala( 1, img_len = win_shape, sep = 0.15, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )
                    
                    plt.subplot(2,2,2)
                    plt.imshow( post_bigwin[exploration:-exploration,exploration:-exploration], cmap = 'Greens', vmin = 100, vmax = 600 )
                    barra_de_escala( 1, img_len = win_shape, sep = 0.15, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )

                    plt.subplot(2,2,3)
                    plt.imshow( pre_win, cmap = 'Reds', vmin = 100, vmax = 600 )
                    plt.imshow( post_bigwin[ exploration - y : -exploration - y , exploration - x : -exploration - x], cmap = 'Greens', vmin = 100, vmax = 600, alpha = 0.6 )
                    barra_de_escala( 1, img_len = win_shape, sep = 0.15, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )

                    plt.subplot(2,2,4)
                    l_cc = len(cross_corr)
                    plt.imshow( np.fliplr(np.flipud(cross_corr)) , cmap = 'Oranges' )
                    plt.plot( [x0],[y0], 'x', c = 'b', ms = 20, label = "Máximo" )
                    if exploration+xo < l_cc and exploration+yo < l_cc:
                        plt.plot( [exploration+xo], [exploration+yo], '+', c = 'k', markersize = 20, label = 'Ajuste' )
                    else: 
                        plt.plot( [exploration - 2],[1], 'x', c = 'r', ms = 20 )
                    plt.plot( [1],[1], 'o', c = marca, ms = 40 )
                    barra_de_escala( 0.5, img_len = l_cc, sep = 0.075, pixel_size = 3/win_shape, font_size = 'xx-large', color = 'w' )
                    # plt.xlim([l_cc-1,0])
                    # plt.ylim([l_cc-1,0])
                    plt.legend()

                    plt.show()
            
            if mode == "Default":
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                
                pre_win_std = np.std( pre_win )
                if pre_win_std > A*pre_std:
                    Y[j,i] = y
                    X[j,i] = x
 
            if mode == "No control":
                y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                y, x = -(y0 - exploration), -(x0 - exploration)
                
                Y[j,i] = y
                X[j,i] = x
            
                    
    deformation_map = Y+translation_Y, X+translation_X       
            
    return deformation_map


def n_iterations( img_post, img_pre, win_shape_0, iterations = 3, exploration = 1000, mode = "Default", A = 0.85, control = [(-1,-1)]):
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
    
    X = "None" #np.zeros([tam0//2, tam0//2])
    Y = "None" #np.zeros([tam0//2, tam0//2])

    mode_array = ["Smooth3"]*(n-1) + [mode]
    control_lista = [ [(-1,-1)] ]*(n-1) + [control]
    for n0 in range(n):
        ventana =  win_shape_0//(2**n0)
        print( n0, ventana )
        Y, X = iteration( img_post, img_pre, ventana, exploration, four_core(Y), four_core(X), mode = mode_array[n0], A = A, control = control_lista[n0] )

    deformation_map = Y, X     

    domain0 = np.arange(Y.shape[0])*ventana + ventana/2
    delta = (len(img_post) - domain0[0] - domain0[-1])/2
    domain = np.meshgrid( domain0 + delta, domain0 + delta )
    
    return domain, deformation_map

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
        # lo cambio por el promedio
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


def traction( Y, X, ps, ws, E, nu, lam, Lcurve = False ):
    """
    Parameters
    ----------
    Y : numpy.2darray
        Deformation map coortinate in pixels.
    X : numpy.2darray
        Deformation map coortinate in pixels.
    ps : float
        nanospheres image's pixel size in meters
    ws : float
        exploration window's size in meters
    E : float
        youngs modulus in Pascals
    nu : float
        posson ratio of the hidrogel
    lam : float
        regularization parameter

    Returns
    -------
    ty : numpy.2darray
        traction cordinate in Pascals
    tx : numpy.2darray
        traction cordinate in Pascals

    """
    l = X.shape[0]

    FuX, FuY = np.fft.fft2( X*ps ), np.fft.fft2( Y*ps ) # FFT de la deformación
    k0 = np.fft.fftfreq(l,ws)*2*np.pi
    k0[0] = 0.00000001 # para evitar dividir por 0
    kx, ky = np.meshgrid( k0, -k0 )
    k = np.sqrt( kx**2 + ky**2 )
    alpha = np.arctan2(ky, kx)

    Kxx = 2*(1+nu)*( (1-nu)+nu*np.sin(alpha)**2 )/(E*k) 
    Kyy = 2*(1+nu)*( (1-nu)+nu*np.cos(alpha)**2 )/(E*k)
    K_ = 2*(1+nu)*( nu*np.cos(alpha)*np.sin(alpha) )/(E*k)  

    # Kxx_inv = E*k*( (1-nu) + nu*np.cos(alpha)**2 )/(1-nu**2)
    # Kyy_inv = E*k*( (1-nu) + nu*np.sin(alpha)**2 )/(1-nu**2)
    # K__inv = -E*k*( nu*np.cos(alpha)*np.sin(alpha) )/(1-nu**2)   

    Ttx, Tty = np.zeros([l]*2), np.zeros([l]*2)
    Ttx_im, Tty_im = np.zeros([l]*2), np.zeros([l]*2)

    for j in range(l):
        for i in range(l):
            TK = np.array( [[ Kxx[j,i]  ,   K_[j,i]  ],
                           [ K_[j,i]   ,   Kyy[j,i] ]]  )
            Tu = np.array(  [ FuX[j,i]   ,  FuY[j,i] ] )
            Id = np.array( [[1,0],[0,1]] )
            # TK_inv = np.linalg.inv(TK)
            M = np.dot( np.linalg.inv( ( np.dot( TK, TK ) + lam*Id ) ) , TK )
            Tt = np.dot( M, Tu )

            Ttx[j,i], Tty[j,i] = np.real(Tt[0]), np.real(Tt[1])
            Ttx_im[j,i], Tty_im[j,i] = np.imag(Tt[0]), np.imag(Tt[1])

    imaginator = np.array( [ [1j]*l ]*l  )
    tx, ty = np.fft.ifft2( Ttx + imaginator*Ttx_im ), np.fft.ifft2( Tty + imaginator*Tty_im )

    # Tvx0 = FuX - Kxx*( Ttx + imaginator*Ttx_im  ) + K_*( Tty + imaginator*Tty_im ) 
    # Tvy0 = FuY - K_*( Ttx + imaginator*Ttx_im  ) + Kyy*( Tty + imaginator*Tty_im )     
    # vx0, vy0 = np.fft.ifft2( Tvx0 ), np.fft.ifft2( Tvy0 )
    
    Tvx0 = Kxx*( Ttx + imaginator*Ttx_im  ) + K_*( Tty + imaginator*Tty_im ) 
    Tvy0 = K_*( Ttx + imaginator*Ttx_im  ) + Kyy*( Tty + imaginator*Tty_im )     
    vx0, vy0 = np.fft.ifft2( Tvx0 ), np.fft.ifft2( Tvy0 )  
  
    if Lcurve == False:
        result = ty, tx    
    else:
        result = ty, tx, vy0, vx0
    
    return result



def Z_iteration0( stack_post0, img_pre0, win_shape, exploration = 1, translation_Y = "None", translation_X = "None", mode = "Smooth3", A = 0.9, z0 = 0 ):
    """
    Parameters
    ----------
    stack_post : numpy.3darray like
        3 dimentional array - images z stack of the nanospheres after removing the cells.
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
    mode : str, optional
        Mode using to calculate maximum correlation. The default is "Default".

    Returns
    -------
    deformation_map : 2 numpy.2darray
        Arrays containing the resulting deformation in Y and X, that is the sum of the previous deformation maps and the calculated position of the cross correlation maximums. 

    """
    l0 = int( len(img_pre0) )
    l = int( len(translation_Y)*win_shape )
    Dl = (l - l0)//2
    img_pre = np.ones( [l]*2 )
    img_pre[ Dl : l0 + Dl, Dl : l0 + Dl ] = img_pre0
    stack_post = np.ones( [len(stack_post0),l,l] )
    stack_post[ :, Dl : l0 + Dl, Dl : l0 + Dl ] = stack_post0
    
    if z0 == 0:
        img_post, ZYX = correct_driff_3D( stack_post, img_pre, 50, info = True)
        z0 = ZYX[0]
    else:
        img_post = correct_driff( stack_post[ z0 ], img_pre, 50)

    
    divis = translation_Y.shape[0]
    Z = np.zeros([divis,divis])
    Y = np.zeros([divis,divis])
    X = np.zeros([divis,divis])

    pre_std = np.std(img_pre)

    for j in range(1, divis -1, 1):
        for i in range(1, divis -1, 1):

            Ay_pre = (j)*win_shape    + (int(translation_Y[j,i]) )
            By_pre = (j+1)*win_shape  + (int(translation_Y[j,i]) )
            Ax_pre = (i)*win_shape    + (int(translation_X[j,i]) )
            Bx_pre = (i+1)*win_shape  + (int(translation_X[j,i]) )

            Ay_post = (j)*(win_shape)   - exploration
            By_post = (j+1)*(win_shape) + exploration
            Ax_post = (i)*(win_shape)   - exploration
            Bx_post = (i+1)*(win_shape) + exploration
            
            pre_win = img_pre[ Ay_pre : By_pre, Ax_pre : Bx_pre ]
            
            # if j == j_p and i == i_p:
            #     plt.figure( tight_layout=True )
            #     plt.subplot(2,2,1)
            #     plt.imshow( pre_win, cmap = cm_crimson, vmin = 100, vmax = 600 )
            #     plt.xticks([])
            #     plt.yticks([])
                
            max_corr = []

            for k in range(-1,2,1):
                post_bigwin = stack_post[ z0 + k , Ay_post : By_post, Ax_post : Bx_post ]
                cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method = "fft") 

                # if j == j_p and i == i_p:
                #     plt.subplot(2,2,k+3)
                #     plt.imshow( post_bigwin[exploration:-exploration,exploration:-exploration], cmap = cm_crimson, vmin = 100, vmax = 600 )
                #     plt.xticks([])
                #     plt.yticks([])
                    
                if mode[:-1] == "Smooth" or mode == "Smooth":
                    ks = 3
                    if mode[-1] != "h":
                        ks = int(mode[-1])
                        
                    cross_corr = smooth( cross_corr , ks )
                    max_corr.append( np.max( cross_corr ) )
                    
                Z[j,i]= np.array(max_corr).argmax() - 1
                    
                
                # if mode == "Fit":
                #     # cross_corr = smooth( cross_corr , 3 )
                #     y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                #     y, x = -(y0 - exploration), -(x0 - exploration)
                #     yo, xo = -y, -x
                    
                #     pre_win_std = np.std( pre_win )
                #     if pre_win_std > A*pre_std:
                #         data = cross_corr
                #         u, v = np.meshgrid(np.linspace(-exploration, exploration, 2*exploration+1), np.linspace(-exploration, exploration, 2*exploration+1) )
                #         amplitude0 = np.max(data)-np.min(data)
                        
                #         popt = [amplitude0, xo, yo, 3, 3, 0, np.min(data)]
                #         popt, pcov = curve_fit_pro(gaussian_2d, (u, v), data.ravel(), p0 = popt )
                #         amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
                        
                #         Y[j,i] = -yo
                #         X[j,i] = -xo
                        
                #         max_corr[k] = amplitude - offset
                    
                #     Z[j,i]= np.array(max_corr).argmax() - 1
                    
    return Z

# en desarrollo
def Z_iteration( stack_post0, img_pre0, win_shape, exploration = 1, translation_Y = "None", translation_X = "None", mode = "Smooth3", A = 0.9, z0 = 0 ):
    """
    Parameters
    ----------
    stack_post : numpy.3darray like
        3 dimentional array - images z stack of the nanospheres after removing the cells.
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
    mode : str, optional
        Mode using to calculate maximum correlation. The default is "Default".

    Returns
    -------
    deformation_map : 2 numpy.2darray
        Arrays containing the resulting deformation in Y and X, that is the sum of the previous deformation maps and the calculated position of the cross correlation maximums. 

    """
    l0 = int( len(img_pre0) )
    l = int( len(translation_Y)*win_shape )
    Dl = (l - l0)//2
    img_pre = np.ones( [l]*2 )
    img_pre[ Dl : l0 + Dl, Dl : l0 + Dl ] = img_pre0
    stack_post = np.ones( [len(stack_post0),l,l] )
    stack_post[ :, Dl : l0 + Dl, Dl : l0 + Dl ] = stack_post0
    
    if z0 == 0:
        img_post, ZYX = correct_driff_3D( stack_post, img_pre, 50, info = True)
    else:
        img_post = correct_driff( stack_post[ z0 ], img_pre, 50)

    divis = translation_Y.shape[0]
    Z = np.zeros([divis,divis])
    Y = np.zeros([divis,divis])
    X = np.zeros([divis,divis])

    pre_std = np.std(img_pre)

    for j in range(1, divis -1, 1):
        for i in range(1, divis -1, 1):

            Ay_pre = (j)*win_shape    + (int(translation_Y[j,i]) )
            By_pre = (j+1)*win_shape  + (int(translation_Y[j,i]) )
            Ax_pre = (i)*win_shape    + (int(translation_X[j,i]) )
            Bx_pre = (i+1)*win_shape  + (int(translation_X[j,i]) )

            Ay_post = (j)*(win_shape)   - exploration
            By_post = (j+1)*(win_shape) + exploration
            Ax_post = (i)*(win_shape)   - exploration
            Bx_post = (i+1)*(win_shape) + exploration
            
            pre_win = img_pre[ Ay_pre : By_pre, Ax_pre : Bx_pre ]
            
            if j == j_p and i == i_p:
                plt.figure( tight_layout=True )
                plt.subplot(2,2,1)
                plt.imshow( pre_win, cmap = cm_crimson, vmin = 100, vmax = 600 )
                plt.xticks([])
                plt.yticks([])
                
            max_corr = np.zeros(3)
            Y_z = np.zeros(3)
            X_z = np.zeros(3)
            
            if z0 == 0:
                z0 = ZYX[0]
                
            for k in range(-1,2,1):
                post_bigwin = stack_post[ z0 + k , Ay_post : By_post, Ax_post : Bx_post ]
                cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method = "fft") 

                if j == j_p and i == i_p:
                    plt.subplot(2,2,k+3)
                    plt.imshow( post_bigwin[exploration:-exploration,exploration:-exploration], cmap = cm_crimson, vmin = 100, vmax = 600 )
                    plt.xticks([])
                    plt.yticks([])
                    
                if mode[:-1] == "Smooth" or mode == "Smooth":
                    ks = 3
                    if mode[-1] != "h":
                        ks = int(mode[-1])
                    cross_corr = smooth( cross_corr , ks )
                    max_corr[ k + 1 ] = np.max( cross_corr )
                    
                    y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                    y, x = -(y0 - exploration), -(x0 - exploration)
                    
                    pre_win_std = np.std( pre_win )
                    if pre_win_std > A*pre_std:
                        Y_z[ k + 1 ] = y
                        X_z[ k + 1 ] = x
                    
                Z[j,i]= max_corr.argmax() - 1
                Y[j,i] = Y_z[max_corr.argmax()]
                X[j,i] = X_z[max_corr.argmax()]
                
                if mode == "Fit":
                    # cross_corr = smooth( cross_corr , 3 )
                    y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                    y, x = -(y0 - exploration), -(x0 - exploration)
                    yo, xo = -y, -x
                    
                    pre_win_std = np.std( pre_win )
                    if pre_win_std > A*pre_std:
                        data = cross_corr
                        u, v = np.meshgrid(np.linspace(-exploration, exploration, 2*exploration+1), np.linspace(-exploration, exploration, 2*exploration+1) )
                        amplitude0 = np.max(data)-np.min(data)
                        
                        popt = [amplitude0, xo, yo, 3, 3, 0, np.min(data)]
                        popt, pcov = curve_fit_pro(gaussian_2d, (u, v), data.ravel(), p0 = popt )
                        amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
                        
                        Y_z[ k + 1 ] = -yo
                        X_z[ k + 1 ] = -xo
                        
                        max_corr[ k + 1 ] = amplitude - offset
                    
                    Z[j,i]= max_corr.argmax() - 1
                    Y[j,i] = Y_z[max_corr.argmax()]
                    X[j,i] = X_z[max_corr.argmax()]
                    
    return Z, Y+translation_Y, X+translation_X



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










