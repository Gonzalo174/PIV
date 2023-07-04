# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:54:48 2023

@author: Usuario
"""

n = 1
pre1 = stack_pre[ n ]
post0 = centrar_referencia( stack_post[ n ] , pre1, 50)
# post0 = centrar_referencia_3D( stack_post, pre1, 50)
# post0, maximo, corrimiento = centrar_referencia( stack_post[ n ] , pre1, 50, maximo = True)



# print(corrimiento)
#%% Ventanas de exploracion

# a, b = 1.5,4.5
a, b = 8, 14
w = 64

pre1_chico = pre1[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post0[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 


plt.figure()
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)


plt.figure()
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow(post0, cmap = 'gray', vmax = 400)
plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )

plt.figure()
plt.title('Pre')
plt.imshow(pre1, cmap = 'gray', vmax = 400)
plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )


# plt.figure()
# plt.title('Trans')
# plt.imshow(celula, cmap = 'gray')
# plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )


#%%
plt.rcParams['font.size'] = 21

borde = 12
pre_win = pre1[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post_bigwin = post0[int(w*a)-borde : int(w*(a+1))+borde, int(w*b)-borde : int(w*(b+1))+borde] 

cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method="fft")
# cross_corr = suavizar( cross_corr, 3 )

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
plt.plot( [borde+x], [y0], 'o',c = 'red', markersize = 10 )
plt.colorbar()

# print( signal.correlate(post0 - post0.mean(), pre1 - pre1.mean(), mode = 'valid', method="fft")[0][0]/(1600**2)  )
# print(np.mean(cross_corr[y0-1:y0+2,x0-1:x0+2])/(w**2))

#%%

pre1_chico_centrado = pre1[ int(w*a + y): int(w*(a+1) + y), int(w*b + x): int(w*(b+1) + x) ]
post0_arriba = centrar_referencia( stack_post[ n-1 ] , pre1, 50)[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 
post0_abajo = centrar_referencia( stack_post[ n+1 ] , pre1, 50)[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 
post0_abajo2 = centrar_referencia( stack_post[ n+2 ] , pre1, 50)[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 





#%%

plt.figure()
plt.title('Post +1')
plt.imshow( post0_arriba , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post -1')
plt.imshow( post0_abajo , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Post -2')
plt.imshow( post0_abajo2 , cmap = 'gray', vmin = 80, vmax = 700)

plt.figure()
plt.title('Pre')
plt.imshow( pre1_chico_centrado , cmap = 'gray', vmin = 80, vmax = 700)



#%%
pre1_chico_centrado = pre1[ int( w*a + y ): int( w*(a+1) + y ), int( w*b + x ): int( w*(b+1) + x ) ]
cc = []
borde = 10

for i in range(6):
    print( i )
    post = centrar_referencia( stack_post[ i ] , pre1, 50)[int(w*a)-borde : int(w*(a+1))+borde, int(w*b)-borde : int(w*(b+1))+borde]
    cross_corr = signal.correlate(post - post.mean(), pre1_chico_centrado - pre1_chico_centrado.mean(), mode = 'valid', method="fft")
    
    plt.figure()
    plt.imshow( post , cmap = 'gray', vmin = 80, vmax = 700)
    
    plt.figure()
    plt.imshow( cross_corr )
    cc.append( np.max(cross_corr) )


plt.plot(cc)















