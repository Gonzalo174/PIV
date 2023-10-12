# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:36:04 2023

@author: Usuario
"""


#%% 20/8

# a, b = 1.5,4.5
a, b = 4, 8
w = 64
a2 = np.mean(pre)/np.mean(post)

pre1_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))]*a2 


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
plt.hist( (val_pre), bins = np.arange(80,700,2), density = True)
plt.title('pre')
plt.ylim([0,0.02])

plt.figure()
plt.hist(val_post, bins = np.arange(80, 700, 2), density = True)
plt.title('post')
plt.ylim([0,0.02])

#%%

a, b = 12, 5
w = 16
a2 = np.mean(pre)/np.mean(post)

pre1_chico = pre[ int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1)) ]
post0_chico = post[int(w*a) : int(w*(a+1)), int(w*b) : int(w*(b+1))]*a2 


plt.figure()
plt.subplot(1,2,1)
plt.title('Pre')
plt.imshow( pre1_chico , cmap = 'gray', vmin = 80, vmax = 700)

plt.subplot(1,2,2)
plt.title('Post')
plt.imshow( post0_chico , cmap = 'gray', vmin = 80, vmax = 700)


val_pre = pre.flatten()
val_pre_chico = pre1_chico.flatten()

plt.figure()
plt.hist( (val_pre), bins = np.arange(80,700,2), density = True)
plt.hist( (val_pre_chico), bins = np.arange(80,700,20), density = True)

plt.title('pre')
plt.ylim([0,0.02])



#%%

inf = 120
a = np.mean(post)/np.mean(pre)

pre_plot = np.copy( (pre+5)*a - inf )
post_plot = np.copy(post - inf )

pre_plot[ pre < 0 ] = 0
pre_plot[ post < 0 ] = 0

c0 = [(0, 0, 0), (0, 0, 0)]
cm0 = ListedColormap(c0)

c1 = []
c2 = []
for i in range(1000):
    c1.append((i/999,0,0))
    c2.append((0,i/999,0))

cm1 = ListedColormap(c1)
cm2 = ListedColormap(c2)


plt.figure()
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 250, alpha = 1)
plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 250, alpha = 0.5)


#%%

from matplotlib.colors import ListedColormap
colors = [(0, 0, 0),  # Black
          (1, 0, 0),  # Red
          (1, 1, 0),  # Yellow
          (0, 1, 0),  # Green
          (0, 0, 1)]  # Blue


#%%
prueba = np.ones([2,2])

prueba[:,0] = [0,0]

plt.imshow(prueba, cmap = cm1, alpha = 1)
plt.imshow(prueba.T, cmap = cm2, alpha = 0.5)


#%% 12/10

val_1 = np.copy(pre[512:].flatten())
val_2 = np.copy(pre[:512].flatten())

plt.figure()
plt.hist( val_2, bins = np.arange(2050)*2, density = False, label = "2")
plt.hist( val_1, bins = np.arange(2050)*2, density = False, label = "1")
plt.yscale("log")
plt.legend()
plt.grid(True)


#%%

um = 800

res = np.zeros( pre.shape )
res[ pre > um ] = 1
pre1 = np.copy( pre )
pre1[ pre > um ] = np.mean( pre*(1-res)*(1024**2)/(1024**2 - np.sum(res)) )

plt.imshow(pre1)

#%%

um = 800

res = np.zeros( post.shape )
res[ post > um ] = 1
post1 = np.copy( post )
post1[ post > um ] = np.mean( post*(1-res)*(1024**2)/(1024**2 - np.sum(res)) )

plt.imshow(post1)









