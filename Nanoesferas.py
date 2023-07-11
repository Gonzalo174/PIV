# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:36:04 2023

@author: Usuario
"""


#%% Ventanas de exploracion

# a, b = 1.5,4.5
a, b = 10, 5
w = 32

pre1_chico = pre1[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post0[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))] 


plt.figure()
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)


plt.figure()
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)

# plt.figure()
# plt.title('Post')
# plt.imshow(post0, cmap = 'gray', vmax = 400)
# plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )

# plt.figure()
# plt.title('Trans')
# plt.imshow(celula, cmap = 'gray')
# plt.plot( [w*b,w*b,w*(b+1),w*(b+1),w*b], [w*a,w*(a+1),w*(a+1),w*a,w*a], linestyle = 'dashed', lw = 3, color = 'r'  )


#%%
plt.rcParams['font.size'] = 21

borde = 5
pre_win = pre1[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post_bigwin = post0[int(w*a)-borde : int(w*(a+1))+borde, int(w*b)-borde : int(w*(b+1))+borde] 

cross_corr = signal.correlate(post_bigwin - post_bigwin.mean(), pre_win - pre_win.mean(), mode = 'valid', method="fft")
# cross_corr = suavizar( cross_corr, 3 )

y0, x0 = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
y, x = -(y0 - borde), -(x0 - borde)

u, v = np.meshgrid(np.linspace(-borde, borde, 2*borde+1), np.linspace(-borde, borde, 2*borde+1))
data = cross_corr.ravel()
initial_guess = [np.max(data)-np.min(data), -x, -y, 3, 3, 0, np.min(data)]
popt, pcov = curve_fit(gaussian_2d, (u, v), data, p0=initial_guess)
amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
gauss_fit = gaussian_2d_plot( (u,v), *popt  )

print(x,y)
print(-xo,-yo)
print( np.mean( cross_corr**2 - gauss_fit**2 ) )

#%%
plt.figure()
plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 + 1 - borde) )
plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 + 1 - borde )
plt.xlabel("Distancia [px]")
plt.ylabel("Distancia [px]")
plt.title("Correlación cruzada")
# plt.text( borde+x, y0+1.2, f'({x},{y})', color='r', weight='bold', ha='center')
plt.imshow( np.flip(cross_corr,1) )# , vmin = 0, vmax = 500000 )
plt.plot( [borde+x], [y0], 'o',c = 'red', markersize = 10 )
plt.plot( [borde-xo], [borde+yo], 'o',c = 'green', markersize = 10 )
plt.colorbar()

# print( signal.correlate(post0 - post0.mean(), pre1 - pre1.mean(), mode = 'valid', method="fft")[0][0]/(1600**2)  )
# print(np.mean(cross_corr[y0-1:y0+2,x0-1:x0+2])/(w**2))

#%%

plt.figure()
plt.yticks( np.arange( borde )*2 +1, -(np.arange( borde )*2 +1- borde) )
plt.xticks( np.arange( borde )*2 +1, np.arange( borde )*2 +1- borde )
plt.xlabel("Distancia [px]")
plt.ylabel("Distancia [px]")
plt.title("Correlación cruzada - Ajuste")

# gauss_fit = gaussian_2d_plot( (u,v), *initial_guess  )

plt.imshow(  np.flip(gauss_fit,1) )
plt.plot( [borde+x], [y0], 'o',c = 'red', markersize = 10 )
plt.plot( [borde-xo], [borde+yo], 'o',c = 'green', markersize = 10 )
plt.colorbar()


#%%

plt.plot( cross_corr[:,-7] )
plt.plot( gauss_fit[:,-7] )



#%% Nanospheres intensity histogram
val_pre = pre.flatten()
val_post = post.flatten()

plt.figure()
plt.hist(val_pre, bins = np.arange(1000))
plt.title('pre')
plt.ylim([0,21000])

plt.figure()
plt.hist(val_post, bins = np.arange(1000))
plt.title('post')
plt.ylim([0,21000])

