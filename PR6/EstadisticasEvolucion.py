
def EstadisticasEvolucion():
	import operator
	import math
	import random

	import numpy

	from deap import algorithms
	from deap import base
	from deap import creator
	from deap import tools
	from deap import gp

	import networkx as nx
	import pandas as pd
	import matplotlib.pyplot as plt
	import numpy as np
	import os

	gen, avg = logbook.select("gen", "avg")
	
	print("El resultado de la evolución es: ")
	print(logbook)
	
	
	# Obtenemos datos
	gen = logbook.select("gen")
	fit_mins = logbook.chapters["fitness"].select("min")
	size_avgs = logbook.chapters["size"].select("avg")


	# Pintamos Fitness mínimo
	fig, ax1 = plt.subplots()
	line1 = ax1.plot(gen, fit_mins, "b-", label="MinimumFitness")
	ax1.set_xlabel("Generation")
	ax1.set_ylabel("Fitness", color="b")
	for tl in ax1.get_yticklabels():
		tl.set_color("b")

	# Pintamos la media del tamaño
	ax2 = ax1.twinx()
	line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
	ax2.set_ylabel("Size", color="r")
	for tl in ax2.get_yticklabels():
		tl.set_color("r")

	lns = line1 + line2
	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc="center")

	plt.plot()
	plt.savefig("regresion.png")

	
	# Extraemos la mejor operación encontrada
	print("\nLa mejor solucion encontrada es: ")
	ffitbest = hof[0]
	print(ffitbest)

	# Extraemos la mejor operación encontrada, fitness mínimo y número de elementos
	for f in hof:
		print(f, evalSymbReg(f, data), len(f))
		
		
	# Hacemos un estudio de una muestra de 10 individuos
	func = toolbox.compile(expr=ffitbest)
	d1 = data.sample(10)
	for j in range(10):
		x = d1.iloc[j]
		#y = func(x.edad,x.peso,x.altura,x.circ_cuello,x.contorno_pecho,x.abdomen,x.caderas,x.muslo,x.rodilla,x.tobillo,x.biceps,x.antebrazo,x.muneca)
		#y = func(x.edad,x.peso,x.altura,x.contorno_pecho,x.abdomen,x.caderas,x.antebrazo)
		y = func(x.edad,x.peso,x.altura)
		print(x.imc, y)
		
	mm=data.median()
	max_values=data.max()
	min_values=data.min()	
		

	# Representamos el ajuste del imc por edades
	edades = np.linspace(min_values.edad,max_values.edad,100)
	#imc = [func(edad,mm.peso,mm.altura,mm.contorno_pecho,mm.abdomen,mm.caderas,mm.antebrazo) for edad in edades]
	imc = [func(edad,mm.peso,mm.altura) for edad in edades]


	x = list(data['edad'])
	y = list(data['imc'])

	plt.scatter(x,y)
	x = sorted(x) 
	plt.plot(edades,imc,"r--") 
	plt.savefig("ajuste_edad.png")

		
	# Representamos el ajuste del imc por altura	
	alturas = np.linspace(min_values.altura,max_values.altura,100)
	#imc = [func(mm.edad,mm.peso,altura,mm.contorno_pecho,mm.abdomen,mm.caderas,mm.antebrazo) for altura in alturas]
	imc = [func(mm.edad,mm.peso,altura) for altura in alturas]


	x = list(data['altura'])
	y = list(data['imc'])

	plt.scatter(x,y)
	x = sorted(x) 
	plt.plot(alturas,imc,"r--") 
	plt.savefig("ajuste_altura.png")


	# Representamos el ajuste del imc por peso
	pesos = np.linspace(min_values.peso,max_values.peso,100)
	#imc = [func(mm.edad,peso,mm.altura,mm.contorno_pecho,mm.abdomen,mm.caderas,mm.antebrazo) for peso in pesos]
	imc = [func(mm.edad,peso,mm.altura) for peso in pesos]


	x = list(data['peso'])
	y = list(data['imc'])

	plt.scatter(x,y)
	x = sorted(x) 
	plt.plot(pesos,imc,"r--") 
	plt.savefig("ajuste_peso.png")

		
		
	# Mostramos el árbol representativo de la función obtenida

	nodes, edges, labels = gp.graph(ffitbest)

	import matplotlib.pyplot as plt
	import networkx as nx

	g = nx.Graph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)


	nx.draw(g, with_labels=True)
	print(labels)

	# Graficamos los valores máximos, mínimos y media
	fig = plt.figure(dpi=100)
	fig.set_figwidth(5)

	#plt.grid()
	plt.title("Valores Máximo, Mínimo y Media")
	#plt.xlabel("---")
	plt.ylabel("Valores")

	plt.plot(mm)
	plt.plot(max_values)
	plt.plot(min_values)

	locs, labels = plt.xticks()
	plt.setp(labels, rotation=90)

	plt.show()
	plt.savefig("arbol_representativo.png")



