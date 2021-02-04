
def DatosFuncion():

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

	nombres_columnas=["edad","peso", "altura", "circ_cuello", "contorno_pecho", "abdomen", "caderas", "muslo", "rodilla", "tobillo", "biceps", "antebrazo", "muneca"]
	data = pd.read_csv("traspuesta.csv", header=0, names=nombres_columnas)
	data2 = pd.read_csv("Y.csv", header=0, names=["imc"])

	# Añadimos el imc del archivo Y a la columna de imc
	data["imc"] = data2


