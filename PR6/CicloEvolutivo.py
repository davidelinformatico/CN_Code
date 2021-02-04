
def CicloEvolutivo():
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


	# Registro de componentes para el algoritmo
	toolbox = base.Toolbox()
	toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)

	toolbox.register("evaluate", evalSymbReg, data=data)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
	toolbox.decorate("mate", gp.staticLimit(key=len, max_value=20))
	toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=20))

	logbook = tools.Logbook()

	def customAlgorithm(pop, toolbox, CXPB, MUTPB, NGEN, stats, halloffame, verbose):
		logbook = tools.Logbook()
		registro=[]
		sin_mejoras = 0
		gen=0
		mejor_fitness = 1e12
		# Evaluate the entire population
		fitnesses = map(toolbox.evaluate, pop)
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit

		while sin_mejoras<NGEN:
			# Select the next generation individuals
			offspring = toolbox.select(pop, len(pop))
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			# Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < CXPB:
					toolbox.mate(child1, child2)
					del child1.fitness.values
					del child2.fitness.values

			for mutant in offspring:
				if random.random() < MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			# The population is entirely replaced by the offspring
			pop[:] = offspring
			
			record = stats.compile(pop)
			logbook.record(gen=gen, **record)

			halloffame.update(pop)
			if verbose:
				print(record)
				#registro.append(record)
			
			gen = gen + 1

			# Si obtenemos un mejor fitness en las estadísticas, reseteamos contador de generaciones sin mejora
			if record['fitness']['min'] < mejor_fitness:
				mejor_fitness = record['fitness']['min']
				sin_mejoras = 0
			else:
				sin_mejoras = sin_mejoras + 1

		return pop, logbook



	random.seed(588962)

	pop = toolbox.population(n=100)
	hof = tools.HallOfFame(1)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", numpy.mean)
	mstats.register("std", numpy.std)
	mstats.register("min", numpy.min)
	mstats.register("max", numpy.max)

	# customAlgorithm(pop, toolbox, CXPB, MUTPB, NGEN, stats, halloffame, verbose)
	pop, logbook = customAlgorithm(pop, toolbox, 0.6, 0.2, 10, stats=mstats, halloffame=hof, verbose=True)


	file = open ("estadisticas_.txt",'w')
	file.write(str(logbook)+ os.linesep)
	file.close()












