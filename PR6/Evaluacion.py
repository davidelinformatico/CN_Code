
def Evaluacion():
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


	def errf(func, x):
		#return (func(x[1].edad,x[1].peso,x[1].altura,x[1].circ_cuello,x[1].contorno_pecho,x[1].abdomen,x[1].caderas,x[1].muslo,x[1].rodilla,x[1].tobillo,x[1].biceps,x[1].antebrazo,x[1].muneca) - x[1].imc)**2
		#return (func(x[1].edad,x[1].peso,x[1].altura,x[1].contorno_pecho,x[1].abdomen,x[1].caderas,x[1].antebrazo) - x[1].imc)**2
		return (func(x[1].edad,x[1].peso,x[1].altura) - x[1].imc)**2



	# Función de evaluación del fitness de los individuos
	# Devuelve el error respecto a la estimación original (mínimos cuadrados)
	def evalSymbReg(individual, data):
		# Transform the tree expression in a callable function
		func = toolbox.compile(expr=individual)
		# Evaluate the mean squared error between the expression
		# and the real function : x**4 + x**3 + x**2 + x
		sqerrors = (errf(func,x) for x in data.iterrows())
		#d1 = data.eval("sqerrors = (imc - func(edad,peso,altura,circ_cuello,contorno_pecho,abdomen,caderas,muslo,rodilla,tobillo,biceps,antebrazo,muneca))**2", engine='python', local_dict={func: func})
		#return d1['sqerrors'].sum()/len(d1)
		return math.fsum(sqerrors) / len(data),
			









