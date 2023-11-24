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
from scipy.stats import ttest_ind as ttest
from scipy.stats import ranksums

#%%
plt.rcParams['figure.figsize'] = [13,8]
plt.rcParams['font.size'] = 22
# plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.family'] = "Yu Gothic"
#%%

df = pd.read_csv( r"C:\Users\gonza\1\Tesis\PIV\celulas 23.11.csv", index_col = 'Celula')
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
conjuntos = [ df.loc[ df["Clase"] == 'MCF10CS' ][key].values, df.loc[ df["Clase"] == 'MCF10SS' ][key].values, df.loc[ df["Clase"] == 'MCF7CS' ][key].values, df.loc[ df["Clase"] == 'MCF7SS' ][key].values ]
plt.rcParams['font.size'] = 11

plt.figure( figsize = [6,3] )
sns.boxplot(  conjuntos, color = colores[0]  )
plt.grid(True)
plt.xticks([0,1,2,3], ['']*4)#['10CS','10SS','7CS','7SS'] )
delta0 = 0.05
plt.text(0-delta0, -0.05, 'MCF10', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(0+delta0, -0.05, 'con suero', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(1-delta0, -0.05, 'MCF10', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(1+delta0, -0.05, 'sin suero', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(2-delta0, -0.05, 'MCF7', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(2+delta0, -0.05, 'con suero', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(3-delta0, -0.05, 'MCF7', rotation=90, va='top', ha='center', color='k', fontsize=11)
plt.text(3+delta0, -0.05, 'sin suero', rotation=90, va='top', ha='center', color='k', fontsize=11)

# plt.xticks([0,1,2,3], ['MCF7SS','MCF7CS','MCF10SS','MCF10CS'])
plt.ylabel( "Deformación promedio [µm]" )
# plt.ylim([-0.01,0.26])
# plt.ylabel( "Deformación/Area [µm]" )
#%%
con = 3
print( ttest( conjuntos[con],conjuntos[0]) )
print( ttest( conjuntos[con],conjuntos[1]) )
print( ttest( conjuntos[con],conjuntos[2]) )
print( ttest( conjuntos[con],conjuntos[3]) )

#%%
con = 2
print( ranksums( conjuntos[con],conjuntos[0]) )
print( ranksums( conjuntos[con],conjuntos[1]) )
print( ranksums( conjuntos[con],conjuntos[2]) )
print( ranksums( conjuntos[con],conjuntos[3]) )

#%%
'''
ttest:
        10CS    10SS     7CS     7SS
10CS     1     0.4198  0.3316  7.1e-3
10SS  0.4198      1    0.6161  1.6e-5
 7CS  0.3316   0.6161     1    9.3e-4
 7SS  7.1e-3   1.6e-5  9.3e-4     1
        
ranksum:
        10CS    10SS     7CS     7SS
10CS     1     0.9423  0.5006  1.5e-4
10SS  0.9423      1    0.4874  5.3e-4
 7CS  0.5006   0.4874     1    2.5e-3
 7SS  1.5e-4   5.3e-4  2.5e-3     1

'''




























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

data.to_csv( "celulas 23.11.csv" )






















