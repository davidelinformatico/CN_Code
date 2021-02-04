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
 
import DatosFuncion
import ConfiguracionProblema
import Evaluacion
import CicloEvolutivo
import EstadisticasEvolucion


DatosFuncion.DatosFuncion()
ConfiguracionProblema.ConfiguracionProblema()
Evaluacion.Evaluacion()
CicloEvolutivo.CicloEvolutivo()
EstadisticasEvolucion.EstadisticasEvolucion()