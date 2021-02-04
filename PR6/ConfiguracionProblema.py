
def ConfiguracionProblema():
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

	def protectedDiv(a, b):
		if b>1e-12: return a/b
		else: return 0

	def protectedSqrt(x):
		if (x>0): return math.sqrt(x)
		else: return 0

	def protectedLog(x):
		if (x>0): return math.log(x)
		else: return 0
		
	def protectedExp(x):
		if (x>10): x=10
		return math.exp(x)

	def square(x):
		return x*x

	def rand101():
		return random.randint(-1,1)



	# Añadimos las operaciones al algoritmo

	pset = gp.PrimitiveSet("MAIN", 3)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(protectedDiv, 2)
	pset.addPrimitive(operator.neg, 1)
	pset.addPrimitive(protectedExp, 1)
	pset.addPrimitive(protectedLog, 1)
	pset.addPrimitive(protectedSqrt, 1)
	pset.addPrimitive(square, 1)
	#pset.addPrimitive(math.cos, 1)
	#pset.addPrimitive(math.sin, 1)
	try:
		pset.addEphemeralConstant("entero", lambda: random.randint(-10,10))
		pset.addEphemeralConstant("rand101", rand101)
		#pset.addEphemeralConstant("k", lambda: random.choice([-1, 0, math.pi, 1, 2, 3, 5, 7]))
	except Exception as e:
		print("-->"+str(e))
		
	pset.addTerminal(math.pi)

	pset.renameArguments(ARG0='edad')
	pset.renameArguments(ARG1='peso')
	pset.renameArguments(ARG2="altura")
	#pset.renameArguments(ARG3="contorno_pecho")
	#pset.renameArguments(ARG4="abdomen")
	#pset.renameArguments(ARG5="caderas")
	#pset.renameArguments(ARG6="antebrazo")
	#pset.renameArguments(ARG7="circ_cuello")
	#pset.renameArguments(ARG8="muslo")
	#pset.renameArguments(ARG9="rodilla")
	#pset.renameArguments(ARG10="tobillo")
	#pset.renameArguments(ARG11="biceps")
	#pset.renameArguments(ARG12="muneca")



	# Evaluación de fitness y generador de individuos

	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)





