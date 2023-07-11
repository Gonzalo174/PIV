# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:54:48 2023

@author: Usuario
"""

plt.rcParams['font.size'] = 21
plt.rcParams['figure.figsize'] = [9,9]


#%% Analize correlation
n = 2
pre = stack_pre[ n ]
post, m, YX = correct_driff( stack_post[ n ] , pre, 50, info = True)
# post = correct_driff_3D( stack_post, pre, 50)

print(YX)
#%% Ventanas de exploracion

# a, b = 1.5,4.5
a, b = 16, 7
w = 32

pre_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 


plt.figure()
plt.title('Pre')
plt.imshow( pre_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow(post, cmap = 'gray', vmax = 400)
plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )

plt.figure()
plt.title('Pre')
plt.imshow(pre, cmap = 'gray', vmax = 400)
plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )

# plt.figure()
# plt.title('Trans')
# plt.imshow(celula, cmap = 'gray')
# plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )


#%%

borde = 10
pre_win = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post_bigwin = post[int(w*a)-borde : int(w*(a+1))+borde, int(w*b)-borde : int(w*(b+1))+borde] 

cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method="fft")
cross_corr = smooth( cross_corr, 3 )

y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
y, x = -(y0 - borde), -(x0 - borde)

print(x,y)

#%%
plt.figure()
plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 + 1 - borde) )
plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 + 1 - borde )
plt.xlabel("Distancia [px]")
plt.ylabel("Distancia [px]")
plt.title("Correlación cruzada")
plt.imshow( np.flip(cross_corr,1) )# , vmin = 0, vmax = 500000 )
plt.plot( [2*borde-x0], [y0], 'o',c = 'red', markersize = 10 )
plt.colorbar()

# print( signal.correlate(post0 - post0.mean(), pre1 - pre1.mean(), mode = 'valid', method="fft")[0][0]/(1600**2)  )
# print(np.mean(cross_corr[y0-1:y0+2,x0-1:x0+2])/(w**2))

#%%

pre_chico_centrado = pre[ int(w*a + y): int(w*(a+1) + y), int(w*b + x): int(w*(b+1) + x) ]

post_arriba = stack_post[ n-1, int(w*a - YX[0]) : int(w*(a+1) - YX[0]), int(w*b - YX[1]) : int(w*(b+1) - YX[1])] 
post_abajo =  stack_post[ n+1, int(w*a - YX[0]) : int(w*(a+1) - YX[0]), int(w*b - YX[1]) : int(w*(b+1) - YX[1])] 
post_abajo2 = stack_post[ n+2, int(w*a - YX[0]) : int(w*(a+1) - YX[0]), int(w*b - YX[1]) : int(w*(b+1) - YX[1])] 

# post_arriba = correct_driff( stack_post[ n-1 ] , pre, 50)[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 
# post_abajo = correct_driff( stack_post[ n+1 ] , pre, 50)[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 
# post_abajo2 = correct_driff( stack_post[ n+2 ] , pre, 50)[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 

#%%

plt.figure()
plt.title('Post +1')
plt.imshow( post_arriba , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post -1')
plt.imshow( post_abajo , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post -2')
plt.imshow( post_abajo2 , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Pre')
plt.imshow( pre_chico_centrado , cmap = 'gray', vmin = 80, vmax = 700)



#%%
# pre1_chico_centrado = pre1[ int( w*a + y ): int( w*(a+1) + y ), int( w*b + x ): int( w*(b+1) + x ) ]
profundidad = []
max_cc = []
borde = 3

for i in range(5):
    print( i )
    post_i = stack_post[ i , int(w*a - YX[0] - borde) : int(w*(a+1) - YX[0] + borde), int(w*b - YX[1] - borde) : int(w*(b+1) - YX[1] + borde)]
    cross_corr_0 = signal.correlate(post_i - post_i.mean(), pre_chico_centrado - pre_chico_centrado.mean(), mode = 'valid', method="fft")
    cross_corr = smooth( cross_corr_0, 3 )
    y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
    
    # plt.figure()
    # plt.imshow( post , cmap = 'gray', vmin = 80, vmax = 700)
    
    plt.figure()
    plt.title( "Profundidad " + str(n-i) )
    plt.imshow( np.flip(cross_corr,1) , vmin = 0, vmax = 2000000 )
    plt.plot( [2*borde-x0], [y0], 'o',c = 'red', markersize = 10 )
    plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 + 1 - borde) )
    plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 + 1 - borde )
    
    profundidad.append( n-i )
    max_cc.append( np.max(cross_corr) )

plt.figure()
plt.plot( profundidad , max_cc )


#%% Reconstruyo con PIV y filtro los datos con, Normalized Median Test (NMT)
vi = 128
it = 3
bordes_extra = 10 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5
modo = "Smooth3"
mapas = False
suave0 = 3

dominio, deformacion = n_iterations( post, pre, vi, it, exploration = bordes_extra, mode = modo)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)

x, y = dominio

#%%
Z = Z_iteration( stack_post, pre, int(vi/(2**(it-1))), edge = 64, plane = n, driff = YX, exploration = 8, translation_Y = Y_nmt, translation_X = X_nmt, mode = "Smooth3")
plt.figure()
plt.imshow(Z)





#%%

def Z_iteration( stack_post, img_pre, win_shape, edge, plane, driff, exploration = 4, translation_Y = "None", translation_X = "None", mode = "Smooth3"):
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
    edge : int, optional
        Maximum exploration. The default is 0.
    mode : str, optional
        Mode using to calculate maximum correlation. The default is "Default".

    Returns
    -------
    deformation_map : 2 numpy.2darray
        Arrays containing the resulting deformation in Y and X, that is the sum of the previous deformation maps and the calculated position of the cross correlation maximums. 

    """
    img_shape = img_pre.shape[0]
    
    img_post0 = correct_driff( stack_post[ plane ] , img_pre, 50)

    cc_area = signal.correlate(img_post0 - img_post0.mean(), img_pre - img_pre.mean(), mode = 'valid', method="fft")/(img_shape**2)
    factor = 0.3
    
    divis = translation_Y.shape[0]
    Z = np.zeros([divis,divis])
    Y = np.zeros([divis,divis])
    X = np.zeros([divis,divis])

    for j in range(divis):
        for i in range(divis):

            Ay_pre = (j)*win_shape    +  edge  + (int(translation_Y[j,i]) + driff[0])
            By_pre = (j+1)*win_shape  +  edge  + (int(translation_Y[j,i]) + driff[0])
            Ax_pre = (i)*win_shape    +  edge  + (int(translation_X[j,i]) + driff[1])
            Bx_pre = (i+1)*win_shape  +  edge  + (int(translation_X[j,i]) + driff[1])

            Ay_post = (j)*(win_shape)   +  edge - exploration
            By_post = (j+1)*(win_shape) +  edge + exploration
            Ax_post = (i)*(win_shape)   +  edge - exploration
            Bx_post = (i+1)*(win_shape) +  edge + exploration
            
            pre_win = img_pre[ Ay_pre : By_pre, Ax_pre : Bx_pre ]
            max_corr = []
            
            # if j == 3 and i == 4:
            #     plt.figure()
                # plt.title('Pre')
                # plt.imshow(pre_win, cmap = "gray")

        
            for k in range(5):
                post_bigwin = stack_post[ plane + k - 2 , Ay_post : By_post, Ax_post : Bx_post ]
                cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method = "fft") 

                # if j == 16 and i == 26:
                #     plt.figure()
                #     plt.title( 'Post' + str(k - 1) )
                #     plt.imshow(post_bigwin[exploration:-exploration,exploration:-exploration], cmap = "gray")

                
                if mode[:-1] == "Smooth" or mode == "Smooth":
                    ks = 3
                    if mode[-1] != "h":
                        ks = int(mode[-1])
                    cross_corr = smooth( cross_corr , ks )
                    
                    # if j == 16 and i == 26:
                    #     plt.figure()
                    #     plt.title('Corr')
                    #     plt.imshow(cross_corr)  
                    
                    y0, x0 = np.unravel_index( cross_corr.argmax(), cross_corr.shape )
                    y, x = -(y0 - exploration), -(x0 - exploration)
                    
                    # cc_area_iteration = cross_corr[y0,x0]/(win_shape**2)
                    # if cc_area_iteration > cc_area*factor:
                    #     Y[j,i] = y
                    #     X[j,i] = x
                        
                    max_corr.append( np.max( cross_corr ) )
                        
                if mode == "Fit":
                    # cross_corr = smooth( cross_corr , 3 )
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
                        
            if j == 16 and i == 26:
                plt.figure()
                plt.title('Z')
                plt.plot( np.arange(5)-2,  max_corr )
                        
            z = np.array(max_corr).argmax() - 2
            Z[j,i] = z
                    
    # deformation_map = Y+translation_Y, X+translation_X       
            
    return Z


#%%


driff = YX
translation_Y, translation_X = deformacion
win_shape = 64
i, j = 3, 4
edge = 64
img_pre = pre

Ay_pre = (j)*win_shape    +  edge  + (int(translation_Y[j,i]) + driff[0])
By_pre = (j+1)*win_shape  +  edge  + (int(translation_Y[j,i]) + driff[0])
Ax_pre = (i)*win_shape    +  edge  + (int(translation_X[j,i]) + driff[1])
Bx_pre = (i+1)*win_shape  +  edge  + (int(translation_X[j,i]) + driff[1])

Ay_post = (j)*(win_shape)   +  edge
By_post = (j+1)*(win_shape) +  edge
Ax_post = (i)*(win_shape)   +  edge
Bx_post = (i+1)*(win_shape) +  edge

pre_win = img_pre[ Ay_pre : By_pre, Ax_pre : Bx_pre ]
post_win = stack_post[ n , Ay_post : By_post, Ax_post : Bx_post ]


plt.figure()
plt.title('Post')
plt.imshow( post_win , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Pre')
plt.imshow( pre_win , cmap = 'gray', vmin = 80, vmax = 700)












