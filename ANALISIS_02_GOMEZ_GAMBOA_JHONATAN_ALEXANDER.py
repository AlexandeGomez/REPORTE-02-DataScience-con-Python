# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 02:40:11 2021

@author: AlexanderG
"""

# =============================================================================
# APARTADO 1: 10 MEJORES RUTAS DE EXPORTACIÓN E IMPORTACIÓN
# =============================================================================
# importando librerias
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from seaborn import clustermap
import matplotlib.pyplot as plt


#-GUARDADO DE LA INFORMACIÓN DEL ARCHIVO CSV Y OBTENIENDO INFO IMPORTANTE-----

# Abriremos el steam o cana del archivo para convertir el iterador "lector"
# en una lista cuya estructura es : 
# 
# data = [[id1,Sentid,Origen,Destino,año,fecha,producto,medio,empresa,valor],
#         [id2,Sentid,Origen,Destino,año,fecha,producto,medio,empresa,valor],
#         [. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ],
#         [idn,Sentid,Origen,Destino,año,fecha,producto,medio,empresa,valor]]
    
with open("synergy_logistics_database.csv","r") as archivo_csv:
    lector = csv.reader(archivo_csv)
    data = list(lector)
del data[0]

# Se obtendra información importante como los años y las rutas
# 
# años = ['2015', '2016', '2017', '2018', '2019', '2020']
#
# rutas = ["Sentido,Origen,Destino,Medio transporte",
#          "Sentido,Origen,Destino,Medio transporte",
#                         . . . . .                 
#          "Sentido,Origen,Destino,Medio transporte"]

años = []
rutas = []
medios = []
paises = []
for line in data:
    if line[4] not in años:
        años.append(line[4])
    if str(line[1])+","+str(line[2])+","+str(line[3])+","+str(line[7]) not in \
        rutas:
        rutas.append(str(line[1])+","+str(line[2])+","+str(line[3])+","+\
                     str(line[7]))
    if line[7] not in medios:
            medios.append(line[7])
    if line[2] not in paises:
        paises.append(line[2])

#--------CREACIÓN DEL REGISTRO CONSIDERANDO LAS CLASIFICACIONES---------------

# Se creara un registro utilizando diccionarios y una lista para clasificar los
# datos en: {año:{ruta:[registro1,registro2,..]}}
#
# registroRutas_a_rutas["año1"]["ruta1"] 
# -> [[id,Sentido1,Origen1,Destino1,año1,fecha,producto,medio1,empresa,valor],
#     [id,Sentido1,Origen1,Destino1,año1,fecha,producto,medio1,empresa,valor],
#     [                             . . . . . .                             ]]

registroRutas_a_ruta = {}
for año in años:
    registroRutas_a_ruta[año]={}
    for ruta in rutas:
        registroRutas_a_ruta[año][ruta]=[]
        for line in data:
            ruta_line = [line[1],line[2],line[3],line[7]]
            if año==line[4] and ruta.split(",")==ruta_line and line[9]!="0":
                registroRutas_a_ruta[año][ruta].append(line)
                
# Se creara un registro utilizando un diccionario cuya clave sera el año y su
# valor sera una lista de la siguiente forma:
#
# registroDemandaRutas_a[año] -> [[ruta1, demanda, año],
#                                 [ruta2, demanda, año],
#                                 [   . . . . . . .   ]]
registroDemandaRutas_a = {}
for año in años:
    registroDemandaRutas_a[año]=[]
    for ruta in rutas:
        registro = registroRutas_a_ruta[año][ruta]
        registroDemandaRutas_a[año].append([ruta,len(registro),año])
    registroDemandaRutas_a[año].sort(reverse=True, key=lambda ls:ls[1])

# rutasDemandas es una lista que contienen las diferentes rutas con sus demandas
# considerando los seis años del conjunto de datos.
#
# rutasDemandas -> [['ruta1', demanda-int],
#                   ['ruta2', demanda-int],
#                   [      . . . . .     ],
#                   ['rutaN', demanda-int]]

rutasDemandas = []
for ruta in rutas:
    cont=0
    for año in años:
        for registro in registroDemandaRutas_a[año]:
            if ruta==registro[0]:
                cont+=registro[1]
    rutasDemandas.append([ruta, cont])
rutasDemandas.sort(reverse=True, key=lambda ls:ls[1])                

print("las mejores 10 rutas considerando demandas son:")
print(rutasDemandas[:10])

# n son las n rutas que se tomaran de cada año para calcular la frecuencia
n = 35

# Se creara una lista cuyos elementos seran listas, cada lista correspondera
# a un registro, corrsponderan a los n mejores de nada año
#
# alld -> [['ruta1', demanda, 'año1'],
#          ['ruta2', demanda, 'año1'],
#          [       . . . . . .      ],
#          ['rutaN', demanda, 'año1'],
#          ['ruta3', demanda, 'año2'],
#          ['ruta4', demanda, 'año2'],
#          [       . . . . . .      ],
#          ['rutaN', demanda, 'año2'],
#          ['ruta5', demanda, 'año3'],
#          ['ruta6', demanda, 'año3'],
#          [       . . . . . .      ],
#          ['rutaN', demanda, 'año3']

alld = []
columns = ["ruta","demanda","año"]
for año in años:
    alld = alld + registroDemandaRutas_a[año][:n]

# Se tomara la lista de listas anterior "alld" y se aplanara para que sea una
# lista de elementos y poder usar el metodo .count() para conocer la frecuencia
# de cada ruta en los 6 años
#
# nd_data al final tendra la siguiente forma:
#
# -> ['ruta1',demanda1,año1,...,rutaA,demanda,año,...,rutaA,demanda,año]

nd_data = np.array(alld)
row, col = nd_data.shape
nd_data = nd_data.reshape(1,row*col)
nd_data = nd_data.tolist()
nd_data = nd_data[0]

# Frecuency_rutas sera un dict al inicio que nos proporcionara la propiedad de los
# conjuntos para evaluar si un elemento existe en el.
# Despues usando la funcion zip() convertiremos el dict en una lista de tuplas
# 
# Frecuency_rutas -> [('ruta1', frecuencia1),
#                     ('ruta2', frecuencia2),
#                     (      . . . . .     ),
#                     ('rutaN', frecuenciaN)]
 
frecuency_rutas = {}
for dts in alld:
    if dts[0] not in frecuency_rutas:
        frecuency_rutas[dts[0]]=nd_data.count(dts[0])
frecuency_rutas = list(zip(frecuency_rutas.keys(),frecuency_rutas.values()))

# Se convertira cada tupla de frecuency_rutas en un diccionario
#
# Frecuency_rutas -> [['ruta1', frecuencia1),
#                     ['ruta2', frecuencia2),
#                     [      . . . . .     ),
#                     ('rutaN', frecuenciaN)]

for i in range(len(frecuency_rutas)):
    frecuency_rutas[i] = list(frecuency_rutas[i])

# a la lista contenida en frecuency_rutas se le agregar al final el total
# de demandas para esa ruta considerando los 6 años
#
# Frecuency_rutas -> [['ruta1', frecuencia1, demandaTotal1),
#                     ['ruta2', frecuencia2, demandaTotal2),
#                     [                . . . . .          ),
#                     ('rutaN', frecuenciaN, demandaTotalN)]
                
for ruta in frecuency_rutas:
    cont = 0
    for año in años:
        for registro in registroDemandaRutas_a[año]:
            if ruta[0]==registro[0]:
                cont+=registro[1]
    ruta.append(cont)
     
# Se ordenan los elementos de frecuency_rutas considerando la demanda total
# con respecto a los 6 años, de mayor a menor
frecuency_rutas.sort(reverse=True, key=lambda ls:ls[1])    

#----------------MAPA DE CALOR Y AGRUPAMIENTO KMEDIAS-------------------------

# Se hara un mapa de calor usando la función clustermap de seaborn
# para ello tomaremos la lista frecuency_rutas en forma dataframe como se muestra:
#
# df ->
#          frecuencia    demanda
# ruta1   frecuencia1   demanda1
# ruta2   frecuencia2   demanda2
#         . . . . . .
# rutaN   frecuenciaN   demandaN
#
# Se van a escalar las entradas utilizando preprocessing.scale()


nd = np.array(frecuency_rutas)

df = pd.DataFrame(data = nd[:,1:].astype(np.float32),
                  index = nd[:,0],
                  columns=["frecuencia","demanda"])

# df_scaled es una copia del dataframe df pero con las entradas escaladas, 
# el mapa de calor nos permite conocer el numero optimo de agrupamientos

df_scaled = df.copy()
df_scaled[["frecuencia","demanda"]]=preprocessing.scale(df_scaled[["frecuencia","demanda"]])

clustermap(df_scaled, cmap="mako")

# Agrupamiento k medias con un agrupamiento de k-esimos grupos
# se usaran dos dataframes uno con las entradas escaladas y otro donde no.
# dataframe nos queda de la siguiente forma:
# 
# df ->        frecuencia   demanda  cluster
#      ruta1  frecuencia1  demanda1    a
#      ruta2  frecuencia2  demanda2    b
#              . . . . . . . . .    
#      rutaN  frecuenciaN  demandaN    n

kmedias = KMeans(7)
kmedias.fit(df_scaled)
clust = kmedias.fit_predict(df_scaled)

df["cluster"]=clust

g = sns.catplot(data=df,
                x = "frecuencia",
                y = "demanda",
                hue="cluster",
                kind="swarm")

# Lineas de codigo que nos permiten extraer filas del dataframe df, considerando
# ciertas condiciones ya sean categoricas o de valores. Con la finalidad de 
# explorar las diferentes rutas.
#
# !URGENTE! Cada que se reinici el programa la enumeración del agrupamiento cambiaran
# por lo cual hay que estar pendiente de que agrupamiento es cual.

indice  = df[df['cluster']==5].index.tolist()
indice2 = df[df['cluster']==2].index.tolist()
indice3 = df[df['cluster']==6].index.tolist()
indice4 = df[df['cluster']==2].index.tolist()

ind2 = df[df['cluster']==4].index.tolist()
ind3 = df[df['demanda']>=135].index.tolist()
set1 = set(ind2).intersection(set(ind3))

# es la forma de mostrar las rutas de cada grupo
indices_all = indice + indice2 + indice3 + indice4 + list(set1)
print(len(indices_all))

# Son 215 rutas totales
# 35 nos dio excelentes resultados
# Linas de codigo que nos permiten graficar en un grafico de barras, cuanto de la
# demanda total es explicada por las rutas propuestas.
#
# A continuación habra dos lineas, si se descomenta la primera y se comenta la segunda
# se graficaran las 35 rutas con mayores demandas en todos los años.
#
# Si se comenta le primera y se descomenta la segunda, nos permitira proponer 
# rutas que no necesariamente tiene que ser las de mayor demandas acumuladas

# lineas de codigo usadas para calcular la demanda total de cada año, la
# demanda del grupo de paises propuesto, y la demanda no explicada por estos

DiezRutas = dict(rutasDemandas[:35])
#DiezRutas = indices_all[:]

set_DiezRutas = set()
for i in DiezRutas:
    set_DiezRutas.add(i)
    
    
bestCountries10_a = {}
for año in años:
    cont=0
    for registro in registroDemandaRutas_a[año]:
        if registro[0] in set_DiezRutas:
            cont+=registro[1]
    bestCountries10_a[año]=cont  
    
otherCountries10_a = {}
for año in años:
    cont=0
    for registro in registroDemandaRutas_a[año]:
        if registro[0] not in set_DiezRutas:
            cont+=registro[1]
    otherCountries10_a[año]=cont
    
allCountries_a = {}
for año in años:
    cont=0
    for registro in registroDemandaRutas_a[año]:
        cont+=registro[1]
    allCountries_a[año]=cont


alld = []
for año in años:
    alld.append([año,"demanda_total",allCountries_a[año]])
    alld.append([año,"demanda_mejores",bestCountries10_a[año]])
    alld.append([año,"demanda_restante",otherCountries10_a[año]])
        
g = sns.catplot(data=pd.DataFrame(alld,columns=["año","grupo","demanda"]),
                x="año",
                y="demanda",
                hue="grupo",
                kind="bar")

# =============================================================================
# APARTADO 2: MEJORES RUTAS DE TRANSPORTE
# =============================================================================

#------------------REGISTRO DE DATOS DEL CSV CLASIFICADOS AÑOS>MEDIOS--------#
# Se creara un registro que leera el archivo csv, clasificara la información en
# años y los diferentes medios de transporte, su estructura es:
#
# registro_mediost_a_mt: {año:{medio_trasporte:[Registro]}}
#
# registro_mediost_a_mt[año][medio] 
# -> [[id1,Sentido,Origen,Destino,Año,Fecha,Producto,Medio,Compañia,Valor],
#     [id2,Sentido,Origen,Destino,Año,Fecha,Producto,Medio,Compañia,Valor],
#     [               . . . . . . . . . . . . . . . . . .               ],
#     [idN,Sentido,Origen,Destino,Año,Fecha,Producto,Medio,Compañia,Valor]]

registro_mediost_a_mt = {}
for año in años:
    registro_mediost_a_mt[año]={}
    for mt in medios:
        registro_mediost_a_mt[año][mt]=[]
        with open("synergy_logistics_database.csv","r") as archivo_csv:
            lector = csv.reader(archivo_csv)
            for ln in lector:
                if ln[4]==año and ln[7]==mt and int(ln[9])!=0:
                    registro_mediost_a_mt[año][mt].append(ln)
                    

#--------------------------VALORES TOTALES DE CADA MEDIO---------------------#
# Se crea concentrado para contar el total de valores movidos por ese medio
# de transporte
# Salida : conteo_mediost_a_mt[año] -> [['Sea', total_valores],
#                                       ['Air', total_valores],
#                                       ['Road',total_valores],
#                                       ['Rail',total_valores]]

conteo_mediost_a_mt = {}
for año in años:
    conteo_mediost_a_mt[año]=[]
    for mt in medios:
        cont=0
        for ram in registro_mediost_a_mt[año][mt]:
            cont+=int(ram[9])
        conteo_mediost_a_mt[año].append([año,cont,mt])


#------COLOCAR LOS DATOS DE VALORES DE CADA MEDIO EN FORMATO PARA DATAFRAME--#
# se crea una lista que contendra todos los Registro de las rutas y su valor
#
# datos_all -> [[2015, valor, Medio transporte1],
#               [2016, valor, Medio transporte1],
#               [          . . . . . .         ],
#               [2020, valor, Medio transporte1],
#               [2015, valor, Medio transporte2],
#               [2016, valor, Medio transporte2],
#               [          . . . . . .         ],
#               [2020, valor, Medio transporte4]]

datos_all=[]
for año in años:
    for cmt in conteo_mediost_a_mt[año]:
        datos_all.append(cmt)
datos_all.sort(key=lambda ls:ls[2])


#--------------------CREACIÓN DE LOS GRAFICOS DE LINEA-----------------------#

# Se crea dataframe para podeer utilizar la función line plot()
# de la siguiente forma:
#
# Esta forma pide un dataframe ordenado pivotado:
# medio         Air         Rail        Road          Sea
# año                                                    
# 2015   7023000000   4437488000  8231077000  11306827000
# 2016   3459111000   5479198000  4710088000  18500041000
# 2017   9065068000   8483234000  2042000000  14838463000
# 2018   2457374000  10632394000  6502062000  17542148000
# 2019   9037995000   4684366000  8732171000  18486953000

datos_pd = pd.DataFrame(datos_all,columns=["año","valores","medio"])
datos_pd.head()
dpd_wide = datos_pd.pivot("año", "medio", "valores")
dpd_wide.head()

# Segunda forma convencional para crear el datafrema y utilizar la función 
# lineplot()
# Esta forma pide un data frame ordenado covencionalmente:
#      año      valores medio
# 0   2015   7023000000   Air
# 1   2016   3459111000   Air
# 2   2017   9065068000   Air
# 3   2018   2457374000   Air
# 4   2019   9037995000   Air
# 5   2020   7219599000   Air

sns.lineplot(data=datos_pd, x="año", y="valores", hue="medio")


# =============================================================================
# ENUNCIADO 3: PAÍSES QUE CONTRIBUYEN AL OCHENTA POR CIENTO
# =============================================================================


# Se obtuvo información relevante como los años y los medios de transporte
# Con la siguiente estructura
#
# años ->  ['2015', '2016', '2017', '2018', '2019', '2020']
# paises -> ['Japan','Germany','China','Italy','USA','Russia','South Korea',
#            'Netherlands','France','Canada','Belgium','Spain','India',
#            'United Kingdom','Australia','Brazil','Switzerland','Mexico',
#            'Austria','Singapore','Vietnam','Malaysia','United Arab Emirates']
años = []
paises = []
for dts in data:
    if dts[4] not in años:
        años.append(dts[4])
    if dts[2] not in paises:
        paises.append(dts[2])
        

# Se creo un concentrado en forma de diccionarios con una lista al final para
#clasificar los registros del csv de la siguiente forma:
# registroPaises_a_o: {año1:{pais_origen:[Registro]}}
#
# registroPaises_a_o[año1][Origen1]
# -> [[id1,Sentid,Origen1,Destino,año1,fecha,producto,medio,empresa,valor],
#     [id2,Sentid,Origen1,Destino,año1,fecha,producto,medio,empresa,valor],
#     [. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ],
#     [idn,Sentid,Origen1,Destino,año1,fecha,producto,medio,empresa,valor]]

registroPaises_a_o = {}
for año in años:
    registroPaises_a_o[año]={}
    for pais in paises:
        registroPaises_a_o[año][pais]=[]
        for dts in data:
            if dts[4]==año and dts[2]==pais and dts[9]!="0":
                registroPaises_a_o[año][pais].append(dts)


# Estructura que calcula el valor total de exportaciones e importaciones para
# cada pais en ese año en especifico cuya estructura es:
#
# valPaises_a_o[año1]-> {pais1: valorTotal1,
#                         pais2: valorTotal2,
#                              . . . . 
#                         paisN: valorTotalN}

valPaises_a_o = {}
for año in años:
    valPaises_a_o[año] = {}
    for pais in paises:
        cont=0
        for r in registroPaises_a_o[año][pais]:
            cont+=int(r[9])
        valPaises_a_o[año][pais] = cont
        
# Considerando los valores totales de cada pais en el diccionaios valPaises_a_o
# calcularemos el 80 del valor de total considerando exportaciones e impotaciones
# de cada año en la el dict porcent80_a el cual tendra la siguiente forma:
#
# porcent80_a -> {'2015': 80%_del_valorTotal,
#                 '2016': 80%_del_valorTotal,
#                 '2017': 80%_del_valorTotal,
#                 '2018': 80%_del_valorTotal,
#                 '2019': 80%_del_valorTotal,
#                 '2020': 80%_del_valorTotal}

porcent80_a = {}
for año in años:
    cont=0
    for pais in valPaises_a_o[año]:
        cont+= valPaises_a_o[año][pais]
    porcent80_a[año]=cont*0.8    

# Se crea un nuevo registro el cual sera un dict, cuya ultima estructura de 
# guardado sera una lista la cual tendra a los paises ordenados de mayor a menor
# considerando el valor total de ese pais, en ese año considerando exp e imp
#
# paisesSorted: {año1: [Registros], año2: [Registros]}
#
# paisesSorted[año1] -> [[pais1, valoresTotales, año1],
#                        [pais2, valoresTotales, año1],
#                        [         . . . . . .       ]] 

paisesSorted = {}
for año in años:
    paisesSorted[año]=[]
    for pais in valPaises_a_o[año]:
        paisesSorted[año].append([pais,valPaises_a_o[año][pais],año])
    paisesSorted[año].sort(reverse=True,key=lambda ls:ls[1])

# Paises80porcet_a es un registro con estructura de dict y una lista como ultimo
# elemento de la jerarquia, cuya función es guardar a los paises que contribuyan
# al 80 % del valor total de las exp e imp.
# llevando un sumatorio y agragando paises siempre que se cumpla:
# ValoresAcumulados <= 80%_del_valorTotal_ese_año
#
# paises80porcent_a: {año1: [Registros], año2: [Registros], ...}
#
# paises80porcent_a[año1] -> [[pais1, valoresTotales, año1],
#                             [pais2, valoresTotales, año1],
#                             [         . . . . . .       ]] 

paises80porcent_a = {}
for año in años:
    paises80porcent_a[año]=[]
    cont=0
    for pval in paisesSorted[año]:
        if cont<porcent80_a[año]:
            paises80porcent_a[año].append(pval)
            cont+=pval[1]
 
# Se crea una lista que contendra los registro de los paises que contribuyen
# al 80 % del valor total de las exportaciones e importaciones, en una sola 
# estructura para su posterior graficación
#
# all -> [[pais1, valoresTotales, 2015],
#         [pais2, valoresTotales, 2015],
#         [         . . . . . .       ],
#         [paisN, valoresTotales, 2020]] 

alld = []
columns = ["pais","valores","año"]
for año in años:
    datos = paises80porcent_a[año][:]
    alld = alld + datos
# Se podra graficar los paises que mayores contribuciones tengan, una grafica
# para cada año
    g = sns.catplot(x="pais",
                    y="valores",
                    data=pd.DataFrame(datos,columns=columns),
                    kind="bar")
    g.set_xticklabels(rotation=-90)


# Grafica frecuencia de cada pais en los 6 años en cuanto a estar entre los
# paises que mas contribuyen
g = sns.catplot(x="pais", 
                kind="count",
                data=pd.DataFrame(alld,columns=columns))
                #height=10, aspect=.7)
g.set_xticklabels(rotation=-90)

# Se aplanara la lista anterior para poder utilizar el metodo de las listas
# .count() y poder encontrar la frecuencia de cada pais en el concentrado alld
#
# datos_tds = [pais1, valoresTotales,año1,pais2, valoresT, año1,...,paisN,valores,añoN]

datos_tds = np.array(alld)
a,b = datos_tds.shape
datos_tds = datos_tds.reshape(1,a*b)
datos_tds = datos_tds[0]
datos_tds = datos_tds.tolist()

# Diccionario creado para contener al pais como clave y como valor su frecuencia
# optenida gracias a la lista datos_tds y el metodo .count()
#
# dicc_paises_cont -> {pais1: frecuencia,
#                      pais2: frecuencia,
#                           . . . .      
#                      paisN: frecuencia}

dicc_paises_cont = {}
for p in alld:
    if p[0] not in dicc_paises_cont:
        dicc_paises_cont[p[0]]=datos_tds.count(p[0])

# El diccionario anterior se convirte a una lista de tuplas usando las funciones
# list() y zip() juntando las claves con los valores
# 
# paises_cont_sorted->[(pais1, frecuencia),
#                      (pais2, frecuencia),
#                           . . . .      
#                      (paisN, frecuencia)]

paises_cont_sorted = list(zip(dicc_paises_cont.keys(),
                              dicc_paises_cont.values()))

# Se tomara la lista anterior de paises_cont_sorted para convertir sus elementos
# tupla en elementos lista para que sea mas facil manipularlos,
#
# paises_cont_sorted->[[pais1, frecuencia],
#                      [pais2, frecuencia],
#                           . . . .      
#                      [paisN, frecuencia]]
for i in range(len(paises_cont_sorted)):
    paises_cont_sorted[i]=list(paises_cont_sorted[i])

# a la lista de listas anteior paises_cont_sorted se le agreagara un nuevo 
# dato a los elementos listas, la cual sera el valor total de todos los años
# a cada pais.
#
# paises_cont_sorted->[[pais1, frecuencia, valorTotal],
#                      [pais2, frecuencia, valorTotal],
#                           . . . .      
#                      [paisN, frecuencia, valorTotal]]

for pcount in paises_cont_sorted:
    cont=0
    for año in años:
        for pais in valPaises_a_o[año]:
            if pcount[0]==pais:
                cont+=valPaises_a_o[año][pais]
    pcount.append(cont)

# Se toma la lista anteior de paises_count_sorted y se ordena usando al función
# .sort() de mayor a menor considerando los valores totales.
paises_cont_sorted.sort(reverse=True,key=lambda ls:ls[2])


# Se crea una nueva lista contenedora para que contendra los registros de unicamente
# los paises contenido en la lista pauses_cont_sorted con la finalidad de graficar
# como se comporta cada a uno a lo largo de los 6 años considerando su valor total
#
# all -> [[pais1, valoresTotales, 2015],
#         [pais2, valoresTotales, 2015],
#         [         . . . . . .       ],
#         [paisN, valoresTotales, 2020]] 

alld =[]
for año in años:
    for pc in paises_cont_sorted:
        for pais in valPaises_a_o[año]:
            if pc[0]==pais:
                alld.append([pais,valPaises_a_o[año][pais],año])
       
# Se graficara la lista alld utilizando la función lineplot() del modulo de
# seaborn, creado un dataframe nuevo de la forma siguiente:
#
# data -> pais                 pais1            pais2  ...           paisN
#               año                                    ...                            
#              2015   valoresTotales   valoresTotales  ...  valoresTotales
#              2016   valoresTotales   valoresTotales  ...  valoresTotales
#                                    . . . . . .       
#              2020   valoresTotales   valoresTotales  ...  valoresTotales

data = pd.DataFrame(alld,columns=["pais","valores","año"])
data = data.pivot("año","pais","valores")
plt.figure(figsize = (10,7))
g = sns.lineplot(data=data)

# Se creara un nuevo dataframe con la finalidad de obtener el mapa de calor
# de los paises que contribuyen al 80% de los valores totales y sus caracteristicas
# utilizando la función clustermap() de seaborn
#
# Se crearon 2 dataframes uno con las entradas escaladas y otros sin escalar
# Se hizo el mapa de calor con las entradas escaladas y el resultado del clustering
# se colocó en el data frame sin escalar

data = np.array(paises_cont_sorted)
indices = data[:,0]
datos = data[:,1:3].astype(np.float32)

datos_scaled = preprocessing.scale(datos)
df_orig = pd.DataFrame(datos,indices,columns=["frecuencia","valores"])
df_scaled = pd.DataFrame(datos_scaled, indices, columns=["frecuencia","valores"])
clustermap(df_scaled, cmap="mako")

# Se hara un agrupamiento k medias con los agrupamientos optimos vistos en el 
# mapa de calor, se hara utilizando la función KMeans del modulo de sklearn
# cuya entrada sera el dataframe con las entradas escaladas y el grafico
# no tendra las salidas escaladas pero el resultado las habra considerado
kmedias = KMeans(3)
kmedias.fit(df_scaled)
clust = kmedias.fit_predict(df_scaled)

df_orig["cluster"]=clust

g = sns.catplot(data=df_orig,
                x="frecuencia",
                y="valores",
                hue="cluster",
                kind="swarm")

# Estas lineas se usan para conocer las lineas del data frame que cumplen
# cierta condicion como lo podria ser pertenecer a un agrupamiento

indice = df_orig[df_orig['cluster']==2]
indice2 = df_orig[df_orig['cluster']==1]

