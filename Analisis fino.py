# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:43:14 2023

@author: gonza
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit as cf
#%%
plt.rcParams['figure.figsize'] = [13,8]
plt.rcParams['font.size'] = 22
# plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.family'] = "Yu Gothic"
#%%

df = pd.read_csv( "celulas 28.10.csv", index_col = 'Celula')
# grupo = df.loc[ df["Clase"] == 'MCF7CS' ]
df["N"] = np.arange(len(df)) + 1
df["N10"] = df["S10"]/df["Area"]
df["DN10"] = df["DS10"]/df["Area"]

#%% Suma/area
plt.errorbar( df["N"] , df["P10"], yerr = df["DP10"], fmt = " ", color = "k"  )
plt.grid( True )
sns.scatterplot( df, x = "N", y = "P10", hue = "Clase" )

#%% Promedio
plt.errorbar( df["N"] , df["N10"], yerr = df["DN10"], fmt = " ", color = "k"  )
plt.grid( True )
sns.scatterplot( df, x = "N", y = "N10", hue = "Clase", marker="o")#, legend = False )

#%% Maximo
plt.grid( True )
sns.scatterplot( df, x = "N", y = "M10", hue = "Clase", marker="o")#, legend = False )


#%%
n0 = []
promediosM = [ ]
DpromediosM = [ ]

for clase in  df["Clase"]:
    grupo = df.loc[ df["Clase"] == clase ]
    promediosM.append( np.mean( grupo["Maximo (10 um)"] ) )
    promediosM.append( np.mean( grupo["Maximo (10 um)"] ) )
    n0.append( np.mean( grupo["N"] ) )
    
#%%
# plt.errorbar( df["Area"] , df["Promedio (10 um)"], yerr = df["Error (10 um)"], fmt = " ", color = "k"  )
sns.scatterplot( df, x = "Promedio (10 um)", y = "Maximo (10 um)", hue = "Clase" )
plt.grid( True )



#%%

key = 'P10'
plt.figure( figsize = [5,5] )
sns.boxplot( [ df.loc[ df["Clase"] == 'MCF10CS' ][key].values, df.loc[ df["Clase"] == 'MCF10SS' ][key].values, df.loc[ df["Clase"] == 'MCF7CS' ][key].values, df.loc[ df["Clase"] == 'MCF7SS' ][key].values ]  )
plt.grid(True)
plt.xticks([0,1,2,3], ['10CS','10SS','7CS','7SS'] )
# plt.xticks([0,1,2,3], ['MCF7SS','MCF7CS','MCF10SS','MCF10CS'])
plt.ylabel( "Deformación promedio [µm]" )
plt.ylim([-0.01,0.26])
# plt.ylabel( "Deformación/Area [µm]" )


#%%
# plt.errorbar( df["N"] , df["N10"], yerr = df["DN10"], fmt = " ", color = "k"  )
plt.grid( True )
sns.scatterplot( df, x = "Area", y = "P10", hue = "Clase", marker="o")#, legend = False )


#%% ajustar linealmente area - prom

def recta(x, m, b):
    return m*x + b

def recta2(x, m):
    return m*x 

colores = { "MCF7SS": "b", "MCF7CS": "orange", "MCF10SS": "g", "MCF10CS": "r"  }

#%%


cla = []
mb = []
Dmb = []

ss7 = df.loc[ df["Clase"] == 'MCF7SS' ]
cs7 = df.loc[ df["Clase"] == 'MCF7CS' ]
df1 = pd.concat( [ss7, cs7] )



for clase in  np.unique(df1["Clase"]):
    grupo = df1.loc[ df1["Clase"] == clase ]
    # if clase == "MCF7CS":
    #     grupo = grupo.drop(["D30_R02"]) 
    # if clase == "MCF10CS":
    #     grupo = grupo.drop(["G18_R25", "G18_R22"])     
        
    X = grupo["Area"].values
    Y = grupo["P10"].values
    DY = grupo["DP10"].values
    
    popt, pcov = cf( recta, X, Y, [1e-4,1], DY )
    # popt, pcov = cf( recta2, X, Y, [1e-4], DY )
    
    cla.append( clase )
    mb.append( popt )
    Dmb.append( np.sqrt( np.diag( pcov ) ) ) 


    X_plot = np.arange(0, 4000, 1 )  
    Y_plot = recta( X_plot, *popt )
    # Y_plot = recta2( X_plot, *popt )
    # plt.figure( )
    # plt.title( clase )
    plt.grid( True )
    # plt.plot( X, Y, "o", c = colores[clase]  )    
    plt.plot( X_plot, Y_plot, c = colores[clase]   )
    
sns.scatterplot( df1, x = "Area", y = "P10", hue = "Clase", s = 200 )#, palette=[(0.333, 0.658, 0.407),(0.768, 0.305, 0.321)])
plt.xlim([-5, 3505])
plt.ylim([-0.03,0.73])

plt.ylabel( "Deformación Promedio [µm]" )
plt.xlabel( "Area [µm²] " )
#%%



















#%%

ss7 = pd.read_csv('data_MCF7SS.csv')
cs7 = pd.read_csv('data_MCF7CS.csv')
ss10 = pd.read_csv('data_MCF10SS.csv') 
cs10 = pd.read_csv('data_MCF10CS.csv') 


data = pd.concat( [ss7, cs7, ss10, cs10] )


data = data.drop( 'Unnamed: 0', axis = 1 )
data = data.set_index('Celula')

data.to_csv( "celulas 28.10.csv" )















#%% Delinear
plt.imshow(mascara)
#%%
mascara0 = smooth(mascara, 3)
mascara1 = np.zeros([1024]*2)
mascara1[mascara0 > 0.5] = 1
#%%
vecinos = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]

# x_borde = [295]
# y_borde = [598]

y_borde = [600]
x_borde = []
j = 0
while len(x_borde) == 0:
    if mascara1[600,j] == 1:
        x_borde.append(j-1)
    j += 1    


# while (y_borde[-1] - y_borde[0])**2 + (x_borde[-1] - x_borde[0])**2 <= 1 and len( x_borde ) > 1000:
for i in range(10000):
    x0 = x_borde[-1] 
    y0 = y_borde[-1]
    for j in range(8):
        v0 = mascara1[ y0 + vecinos[j-1][0], x0 + vecinos[j-1][1] ]
        v1 = mascara1[   y0 + vecinos[j][0],   x0 + vecinos[j][1] ]
        if v0 == 0 and v1 == 1:
            x_borde.append( x0 + vecinos[j-1][1] )
            y_borde.append( y0 + vecinos[j-1][0] )
    

#%%

plt.imshow( celula_pre, cmap='gray' )
plt.plot(x_borde,y_borde,c = 'k', linestyle = (1, (10, 50)), linewidth = 2 )


#%%
# plt.imshow(mascara, cmap='gray', vmin = -1, vmax = 2)
plt.plot(x_borde,y_borde,c = 'k', linestyle = (1, (10, 500)), linewidth = 2 )
# plt.xlim([500,700])
# plt.ylim([100,350])







