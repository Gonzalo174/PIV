# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:43:14 2023

@author: gonza
"""


df = pd.read_csv( "celulas 25.10.csv", index_col = 'Celula')
grupo = df.loc[ df["Clase"] == 'MCF10CS' ]















#%%

ss7 = pd.read_csv('dataMCF7SS.csv')
cs7 = pd.read_csv('dataMCF7CS.csv')
ss10 = pd.read_csv('dataMCF10SS.csv') 
cs10 = pd.read_csv('dataMCF10CS.csv') 


data = pd.concat( [ss7, cs7, ss10, cs10] )


data = data.drop( 'Unnamed: 0', axis = 1 )
data = data.set_index('Celula')

data.to_csv( "celulas 25.10.csv" )




