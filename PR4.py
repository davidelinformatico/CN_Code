# Importamos
import random, os
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

#------------------------------------------------------#
#------------Parámetros de configuración:--------------#
#------------------------------------------------------#

nombreArchivo = "d_pet_pictures.txt" # Cambiar a parámetro
archivoSalida = "solucion.txt" 
pesos = (1.0, 1.0)

poblacionInicial = 12
numeroIndividuos = 24
seleccionNMejores = 6

probabilidadMutacion = 0.2
probabilidadCruce = 0.82
numeroGeneraciones = 20

#------------------------------------------------------#
#-----------------------Código:------------------------#
#------------------------------------------------------#

# Leemos el archivo
f = open(nombreArchivo)
data = f.read()
f.close()

# Generamos una matriz cortando por los saltos de línea
matrix1 = data.split('\n')

# Obtenemos el número de líneas
lineas=int(matrix1[0])

# Obviamos la primera línea
matrizSinCabecera = matrix1[1:]

# Generamos la matriz limpia
matriz=[]
for i in range(lineas):
    matriz.append([i] + matrizSinCabecera[i].split(' '))

# Extraemos los tags de cada uno de los elementos.
tags = []

for i in range(lineas):
    tags.append(set(matriz[i][3:]))

toolbox = base.Toolbox()
# More info: https://deap.readthedocs.io/en/master/api/tools.html#module-deap.tools

#creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("FitnessMax", base.Fitness, weights=pesos)
# Los individuos se muestran como una lista y la maximización del fitness
creator.create("Individual", list, fitness=creator.FitnessMax) 

horizontales = list(filter(lambda r:r[1]=='H', matriz))
verticales = list(filter(lambda r:r[1]=='V', matriz))

#Individuo de ejemplo:
individuo = [(1,2),(3,),(0,)]

# Individuos
def initIndividual(icls, content):
    return icls(content)
# Población
def initPopulation(pcls, ind_init, lineas, n):
    aleatorios = []
    for k in range(n):
        ind = [(x[0],) for x in horizontales]
        random.shuffle(verticales)
        ind = ind + [(x[0][0], x[1][0]) for x in zip(verticales[::2],verticales[1::2])]
        random.shuffle(ind)
        aleatorios.append(ind)
    return pcls(ind_init(c) for c in aleatorios)

#Tendrá tantos atributos como líneas tenga el archivo
toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", initPopulation, list, toolbox.individual, lineas)

# Se genera un único individuo
ind=toolbox.individual(individuo)

# Se inicializa la poblacion. Tendrá un total de 'n' individuos. 
# Se genera como una lista de individuos 
population = toolbox.population(n=numeroIndividuos)

# Se imprime la población: 24 individuos, con un número de genes como líneas tengamos en nuestro archivo
#print("Poblacion: ",population)
population

def feno(individuo):
    todas = []
    vistas = set([])
    
    for t in individuo:
        r = set([])
            
        for j in list(t):
            if j in vistas: return None
            vistas.add(j)
            r = r.union(tags[j])
        todas.append(r)
        
        if len(t)==1 and matriz[t[0]][1]=='H': continue
        elif len(t)==2 and matriz[t[0]][1]=='V' and matriz[t[1]][1]=='V': continue
        else: return None
        
    return todas

# Ésta es nuestra función de fitness:
def adaptacion(individuo):
    transiciones = feno(individuo)
    if not transiciones: return (-1,) # penalización
    
    factorInteres=[]
    for i in range(len(transiciones)-1):
        # Generamos las intersecciones
        s1=set(transiciones[i])
        s2=set(transiciones[i+1])
        s3=s1.difference(s2)
        s4=s2.difference(s1)
        aux=s1.intersection(s2)
        #print(s1,s2,s3,s4,aux)
        # Obtenemos las longitudes mínimas de los sets
        factorInteres.append(min(len(s3),len(s4),len(aux)))

    return (sum(factorInteres),)

#---------------------------------------------------------------------------------------------------------------------
# Se seleccionan procedimiento standard para cruce, mutacion y seleccion
toolbox.register("select", tools.selTournament, tournsize=6) # Seleccion de los n mejores
toolbox.register("mate", tools.cxOnePoint) # Cruce
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.7) # Mutacion. Cambiamos la probabilidad en indpb

# Se define cómo se evaluará cada individuo
# En este caso, se hará uso de la función de evaluación "adaptacion"
toolbox.register("evaluate", adaptacion)

#---------------------------------------------------------------------------------------------------------------------
stats = tools.Statistics(lambda ind: ind.fitness.values) 
stats.register("avg", np.mean) 
stats.register("std", np.std) 
stats.register("min", np.min) 
stats.register("max", np.max) 


#---------------------------------------------------------------------------------------------------------------------
# Se genera una población inicial.
population = toolbox.population(n=poblacionInicial)

# Se llama al algoritmo que permite la evolucion de las soluciones
population, logbook = algorithms.eaSimple(population, toolbox, 
                                    cxpb=probabilidadCruce, mutpb=probabilidadMutacion, # Probabilidades de cruce y mutacion
                                    ngen=numeroGeneraciones, verbose=False, stats=stats) # Numero de generaciones a completar y estadisticas a recoger
# Por cada generación, la estructura de logbook va almacenando un resumen de los
# avances del algoritmo.


file = open ("estadisticas.txt",'a')
file.write(str(logbook)+ os.linesep)
file.close()

file = open ("solucion.txt",'a')
file.write(str(len(tools.selBest(population,1)[0]))+ os.linesep)
salida=set(tools.selBest(population,1)[0])
a=0
for i in salida:
    if a==0:
        if len(i)>1:
            file.write(str(i)[1:-1])
        else:
            file.write(str(i)[1:-2])
        a=1    
    else:
        if len(i)>1:
            file.write(os.linesep+str(i)[1:-1])
        else:
            file.write(os.linesep+str(i)[1:-2])
        
file.close()

# Se recuperan los datos desde el log
gen = logbook.select("gen")
avgs = logbook.select("avg")
    
# Se establece una figura para dibujar
fig = plt.figure()
    
# Se representa la media del valor de fitness por cada generación
ax1 = plt.gca()
line1 = ax1.plot(gen, avgs, "r-", label="Average Fitness")    
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")

#plt.plot()
plt.draw()
plt.savefig("cómputo.png")


