# Importamos
import random, os, math
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

#------------------------------------------------------#
#------------Parámetros de configuración:--------------#
#------------------------------------------------------#

#nombreArchivo = "b_lovely_landscapes_mo.txt"
#nombreArchivo = "c_memorable_moments_mo.txt"
nombreArchivo = "d_pet_pictures_mo.txt"
#nombreArchivo = "e_shiny_selfies_mo.txt"

pesos = (1.0, 1.0)

poblacionInicial = 12
numeroIndividuos = 24
seleccionNMejores = 6

probabilidadMutacion = 0.21
probabilidadCruce = 0.82
numeroGeneraciones = 200

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
lineasTotales=int(matrix1[0])

# Obviamos la primera línea
matrizSinCabecera = matrix1[1:]

# Generamos la matriz limpia incluyendo la restricción necesaria
# Si tenemos más de 5K fotos, contemplamos únicamente las 5K primeras
lineas = int(lineasTotales/2)
if lineas > 5000: lineas = 5000

matriz=[]
for i in range(lineas):
    matriz.append([i] + matrizSinCabecera[i].split(' '))


# Extraemos los tags de cada uno de los elementos.
tags = []

for i in range(lineas):
    tags.append(set(matriz[i][4:]))

toolbox = base.Toolbox()
# More info: https://deap.readthedocs.io/en/master/api/tools.html#module-deap.tools

#creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("FitnessMax", base.Fitness, weights=pesos)
# Los individuos se muestran como una lista y la maximización del fitness
creator.create("Individual", list, fitness=creator.FitnessMax) 

horizontales = list(filter(lambda r:r[1]=='H', matriz))
verticales = list(filter(lambda r:r[1]=='V', matriz))


# Individuos
def initIndividual(icls, content):
    return icls(content)

# Población
# n=número de individuos en la población
def initPopulation(pcls, ind_init, lineas, n=20):
    aleatorios = []
    for k in range(n):
        ind = [(x[0],) for x in horizontales]
        random.shuffle(verticales)
        ind = ind + [(x[0][0], x[1][0]) for x in zip(verticales[::2],verticales[1::2])]
        random.shuffle(ind)
        ind2 = ind[:lineas]
        aleatorios.append(ind2)
    return pcls(ind_init(c) for c in aleatorios)


#Tendrá tantos atributos como líneas tenga el archivo
toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", initPopulation, list, toolbox.individual, lineas)

# Generamos la población inicial
population = toolbox.population(n=64)

# generamos un individuo
individuo = population[0]

# Se genera un único individuo
ind=toolbox.individual(individuo)
#print("Individuo: len=", len(ind))

def feno(individuo):
    todas = []
    vistas = set([])
    
    for t in individuo:
        r = set([])
            
        for j in list(t):
            #if j in vistas: return None
            vistas.add(j)
            r = r.union(tags[j])
        todas.append(r)

        if len(t)==1 and matriz[t[0]][1]=='H': continue
        elif len(t)==2 and matriz[t[0]][1]=='V' and matriz[t[1]][1]=='V': continue
        else: return None
        
    return todas

def calcularTamanos(individuo):
    todas = []
    
    for t in individuo:
        todas.append(sum([float(matriz[j][2]) for j in t]))
        
    return todas



# Ésta es nuestra función de fitness:
def adaptacion(individuo):
    transiciones = feno(individuo)
    if not transiciones: return (-1,0) # penalización
    
    # calcular peso total
    pesosIndividuo=calcularTamanos(individuo)
    pesoTotal = sum(pesosIndividuo)

    # obtener slice con la quinta primera parte de los tamaño
    slice=int(math.ceil(pesoTotal*0.2))
    
    # Obtener 20% del tamaño del individuo
    #pesoRecortado=math.ceil(len(pesosIndividuo)*0.2)
    pesoRecortado=math.ceil(len(pesosIndividuo)*0.2)
        
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
        
    # si la suma del slice excede 0.2*tamaño total => return (-1, 0)
    indRecortado=pesosIndividuo[:pesoRecortado]
    #if sum(indRecortado)>pesoTotal*0.2: return (sum(factorInteres), -1)
    if sum(indRecortado)>pesoTotal*0.2: pesoTotal=pesoTotal*5
    
    return (sum(factorInteres),1.0/pesoTotal if pesoTotal>0 else 0)

#---------------------------------------------------------------------------------------------------------------------
# Se seleccionan procedimiento standard para cruce, mutacion y seleccion
toolbox.register("select", tools.selSPEA2) # Seleccion de los n mejores
toolbox.register("mate", tools.cxOnePoint) # Cruce
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=probabilidadMutacion) # Mutacion. Cambiamos la probabilidad en indpb

# Se define cómo se evaluará cada individuo
# En este caso, se hará uso de la función de evaluación "adaptacion"
toolbox.register("evaluate", adaptacion)

#---------------------------------------------------------------------------------------------------------------------
stats_fit = tools.Statistics(lambda ind: (ind.fitness.values[0] if ind.fitness.values else None)) 
stats_size= tools.Statistics(lambda ind: (ind.fitness.values[1] if ind.fitness.values else None)) 

mstats= tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean) 
mstats.register("std", np.std) 
mstats.register("min", np.min) 
mstats.register("max", np.max) 


#---------------------------------------------------------------------------------------------------------------------
# Se genera una población inicial.
population = toolbox.population(n=poblacionInicial)

'''Se ha llamado al nuevo objeto multiestadístico'''
# Se llama al algoritmo que permite la evolucion de las soluciones
population, logbook = algorithms.eaSimple(population, toolbox, 
                                    cxpb=probabilidadCruce, mutpb=probabilidadMutacion, # Probabilidades de cruce y mutacion
                                    ngen=numeroGeneraciones, verbose=False, stats=mstats) # Numero de generaciones a completar y estadisticas a recoger
# Por cada generación, la estructura de logbook va almacenando un resumen de los
# avances del algoritmo.


file = open ("estadisticas_"+nombreArchivo+".txt",'a')
file.write(str(logbook)+ os.linesep)
file.close()

file = open ("fenotipo_"+nombreArchivo+".txt",'a')
file.write(str(feno(tools.selBest(population,1)[0]))+ os.linesep)
file.close()

file = open ("solucion_"+nombreArchivo+".txt",'a')
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
fit_avgs = logbook.chapters["fitness"].select("avg")
size_avgs = logbook.chapters["size"].select("avg")
    
# Se establece la figura a dibujar
fig, ax1 = plt.subplots()

# Se representa la media de los valores de fitness y peso por cada generación
line1 = ax1.plot(gen, fit_avgs, "b-", label="Average Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
ax2.set_ylabel("Size", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

#plt.show()
plt.draw()
plt.savefig("cómputo_"+nombreArchivo+".png")


