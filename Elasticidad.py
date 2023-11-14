# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:25:59 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import oiffile as of


#%%
plt.rcParams['figure.figsize'] = [13,8]
plt.rcParams['font.size'] = 22
plt.rcParams['font.family'] = "Times New Roman"

#%%

E3_0 = np.loadtxt(r"C:\Users\gonza\1\Tesis\2023\Mediciones de elasticidad Gel B\E3_values.txt") 
E4_0 = np.loadtxt(r"C:\Users\gonza\1\Tesis\2023\Mediciones de elasticidad Gel B\E4_values.txt") 
E5_0 = np.loadtxt(r"C:\Users\gonza\1\Tesis\2023\Mediciones de elasticidad Gel B\E5_values.txt") 
E6_0 = np.loadtxt(r"C:\Users\gonza\1\Tesis\2023\Mediciones de elasticidad Gel B\E6_values.txt") 


E3 = np.reshape( E3_0, [16,16])
E4 = np.reshape( E4_0, [16,16])
E5 = np.reshape( E5_0, [16,16])
E6 = np.reshape( E6_0, [16,16])
# E6[-1,-1] = 0

E1 = np.concatenate( (E5,E4), axis = 1 )
E2 = np.concatenate( (E3,E6), axis = 1 )
E0 = np.concatenate( (E2,E1), axis = 0 )


#%%

delta = 4.5
for i in range(1,31):
    for j in range(1,31):
        diff = E0[j,i] - 32
        if np.abs(diff) > delta:
            E0[j,i] = np.mean( E0[j-1:j+2, i-1:i+2] )

#%%
scale_length = 250  # Length of the scale bar in pixels
scale_pixels = scale_length/(1000/16)
scale_unit = 'nm'  # Unit of the scale bar

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
E0[0,0] = 36.001

#%%

start_x = 26  # Starting x-coordinate of the scale bar
start_y = 3  # Starting y-coordinate of the scale bar

plt.figure( figsize = [15,8], tight_layout=False)
plt.subplot(1,2,1 )
plt.imshow(E0)#s[1:-1,1:-1])
plt.colorbar( location = "bottom", label = 'Módulo de Young [kPa]', pad = 0.01, fraction = 0.045)#, ticks = [20,26,28,30,32,34,36] )  #, shrink = 0.5#, )
plt.xticks([])
plt.yticks([])
for i in np.arange(-0.3, 0.3, 0.01):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i, start_y + i], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y-1, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "large")

plt.subplot(1,2,2)
plt.hist( E0.flatten(), bins = np.arange(26.5,37.5,1), color = cm.viridis(0.6) )
plt.xlabel('Módulo de Young [kPa]')
plt.ylabel("Cuentas")
plt.xlim([27,37])
plt.grid(True)
plt.show()
#%%
E0s = smooth(E0, 3)
plt.hist( E0.flatten(), bins = np.arange(26.5,37.5,1) )
print(np.round( np.mean(E0), 1) , np.round( np.std(E0), 1 ) )



#%%

stack1 = of.imread( r"C:\Users\gonza\1\Tesis\2023\Mediciones de elasticidad Gel B\24-pasogrueso-gel_I.oif" )
stack2 = of.imread( r"C:\Users\gonza\1\Tesis\2023\Mediciones de elasticidad Gel B\C1-03-10X-dz1-pw15.oif" )


plt.plot( np.arange(len(stack1[0]) )*2, np.mean( np.mean(stack1[0], axis = 1 ), axis = 1 )  )
# plt.plot( np.arange(len(stack2[0]) ), np.mean( np.mean(stack2[0], axis = 1 ), axis = 1 )  )

#%%
plt.figure( figsize = [16,6] )
plt.plot( -(np.arange(len(stack1[0]) )*2 - 26), np.mean( np.mean(stack1[0], axis = 1 ), axis = 1 ), 'o', c = 'k'  )
plt.plot( [0]*2, [80,920], ls = '--', lw = 3, c = (0.768, 0.305, 0.321), label = 'medio-gel'   )
plt.plot( [-105.3]*2, [80,920], ls = '--', lw = 3, c = (0.768, 0.305, 0.321), label = 'gel-vidrio'   )
plt.grid()
# plt.legend()
plt.ylabel('Intensidad [U.A.]')
plt.xlabel('Profundidad [µm]')
plt.xlim([30,-150])
plt.ylim([80,920])

#%%
plt.plot( np.arange(len(stack1) )*2, np.mean( np.mean(stack1[:,:256,:256], axis = 1 ), axis = 1 )  )
plt.plot( np.arange(len(stack1) )*2, np.mean( np.mean(stack1[:,:256,256:512], axis = 1 ), axis = 1 )  )
plt.plot( np.arange(len(stack1) )*2, np.mean( np.mean(stack1[:,256:512,:256], axis = 1 ), axis = 1 )  )
plt.plot( np.arange(len(stack1) )*2, np.mean( np.mean(stack1[:,256:512,256:512], axis = 1 ), axis = 1 )  )

#%%
plt.imshow( stack1[0,17], cmap = "gray")#, 480:, 480:] )
# plt.imshow( stack1[1,66], cmap = "gray")#, :50, :50] )

#%%
for i in range(len(stack1)):
    plt.figure()
    plt.title(str(i))
    plt.imshow( stack1[i], cmap = 'gray', vmin = 70, vmax = 800 )


