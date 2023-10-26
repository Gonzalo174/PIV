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
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = "Times New Roman"

#%%

df = pd.read_csv( "celulas 25.10.csv", index_col = 'Celula')
grupo = df.loc[ df["Clase"] == 'MCF7CS' ]
df["N"] = np.arange(len(df)) + 1
df["Normalizada"] = df["Promedio (10 um)"]/df["Area"]
#%%
plt.errorbar( df["N"] , df["Promedio (10 um)"], yerr = df["Error (10 um)"], fmt = " ", color = "k"  )
plt.grid( True )
sns.scatterplot( df, x = "N", y = "Promedio (10 um)", hue = "Clase" )
# sns.scatterplot( df, x = "Area", y = "Maximo (10 um)", hue = "Clase", marker="X", legend = False )
# plt.xlim([0, 4000])

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


for clase in  np.unique(df["Clase"]):
    grupo = df.loc[ df["Clase"] == clase ]
    if clase == "MCF7CS":
        grupo = grupo.drop(["D30_R02"]) 
    if clase == "MCF10CS":
        grupo = grupo.drop(["G18_R25", "G18_R22"])     
        
    X = grupo["Area"].values
    Y = grupo["Promedio (10 um)"].values
    DY = grupo["Error (10 um)"].values
    
    popt, pcov = cf( recta2, X, Y, [1e-4], DY )
    cla.append( clase )
    mb.append( popt )
    Dmb.append( np.sqrt( np.diag( pcov ) ) ) 


    X_plot = np.arange(100, 4000, 1 )  
    Y_plot = recta2( X_plot, *popt )
    # plt.figure( )
    # plt.title( clase )
    plt.grid( True )
    # plt.plot( X, Y, "o", c = colores[clase]  )    
    plt.plot( X_plot, Y_plot, c = colores[clase]   )
    

sns.scatterplot( df, x = "Area", y = "Promedio (10 um)", hue = "Clase" )
plt.xlim([0, 4000])



#%%

ss7 = pd.read_csv('dataMCF7SS.csv')
cs7 = pd.read_csv('dataMCF7CS.csv')
ss10 = pd.read_csv('dataMCF10SS.csv') 
cs10 = pd.read_csv('dataMCF10CS.csv') 


data = pd.concat( [ss7, cs7, ss10, cs10] )


data = data.drop( 'Unnamed: 0', axis = 1 )
data = data.set_index('Celula')

data.to_csv( "celulas 25.10.csv" )




