#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================
# Copyright © 2024 Intel Corporation
# 
# SPDX-License-Identifier: MIT
# =============================================================


# # Genetic Algorithms on GPU using Intel Distribution of Python numba-dpex
# 
# This code sample shows how to implement a basic genetic algorithm with Data Parallel Python using numba-dpex.
# 
# ## Genetic algorithms
# 
# Let's start with the question **What is a genetic algorithm?**. It is an algorithm, search heuristic inspired by the process of natural selection. It is usually applied to various optimization problems, NP-hard problems for which finding a solution by standard methods is very time and resource consuming. This algorithm makes it possible to obtain a satisfying high quality result based on biology-inspired operations, such as:
#  
# * selection - is the process of selecting parents who mate and recombine to create offspring for the next generation. Parent selection is very crucial to the convergence rate of the GA as good parents drive individuals to better and fitter solutions.
# * crossover - is a process similar to biological crossover. In this, more than one parent is selected and one or more offspring are produced using the genetic material of the parents.
# * mutation - small random tweak in the chromosome, to get a new solution. It is used to maintain and introduce diversity in the genetic population and is usually applied with a low probability. 
# 
# To apply the genetic algorithm to a specific problem, it is important to define the representation of the chromosome, as well as how the three operations should look like. 
# 
# In this example, we will show, first, the general implementation of the genetic algorithm, and then the adaptation of this function to the Traveling Salesman Problem.

# Let us start with import of the libraries used in this code sample.

# In[ ]:


import numpy as np
import time
import random
import math


# ## Initialize population
# 
# Then, we can initialize the population. In this code sample, we have a population of size 5000, a chromosome size is 10, and there will be 5 generations. 
# Each chromosome will contain 10 random floats between 0 and 1.
# 
# We also are setting a seed to be able to reproduce the results later.

# In[ ]:


random.seed(1111)

pop_size = 5000
chrom_size = 10
num_generations = 5

fitnesses = np.zeros(pop_size, dtype=np.float32)
chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
for i in range(pop_size):
  for j in range(chrom_size):
    chromosomes[i][j] = random.uniform(0,1) #random float between 0.0 and 1.0


# ## Genetic Algorithm implementation
# 
# The next step is to create a general purpose genetic algorithm, which means calculating the fitness value for all the chromosomes, selection of chromosomes, crossover and mutation functions.
# 
# ### Simple evaluation method
# 
# We are starting with a simple genome evaluation function. This will be our baseline and comparison for numba-dpex.
# In this example, the fitness of an individual is computed by an arbitrary set of algebraic operations on the chromosome.

# In[ ]:


def eval_genomes_plain(chromosomes, fitnesses):
  for i in range(len(chromosomes)):
    num_loops = 3000
    for j in range(num_loops):
      fitnesses[i] += chromosomes[i][1]
    for j in range(num_loops):
      fitnesses[i] -= chromosomes[i][2]
    for j in range(num_loops):
      fitnesses[i] += chromosomes[i][3]

    if (fitnesses[i] < 0):
      fitnesses[i] = 0


# ### Crossover
# 
# The crossover operation creates children's genomes from selected parent chromosomes. As shown in the figure below, in this sample, the one-point crossover is made and one children's genome is created.
# 
# The first part of the child's genome comes from the first parent, and the second half, from the second parent.
# 
# <img src="./assets/crossover.png" alt="image" width="auto" height="400">
# 

# In[ ]:


def crossover(first, second):
  index = random.randint(0, len(first) - 1)
  index2 = random.randint(0, len(second) - 1)

  child_sequence = []

  for y in range(math.floor(len(first) / 2)):
      child_sequence.append( first[ (index + y) % len(first) ] )

  for y in range(math.floor(len(second)/ 2)):
      child_sequence.append( second[ (index2 + y) % len(second) ] )
      
  return child_sequence


# ### Mutation
# 
# The mutation operation can change the chromosome, as shown in the figure. In this code sample, there is a 1% chance of a random mutation.
# 
# <img src="./assets/mutation.png" alt="image" width="auto" height="300">
# 

# In[ ]:


def mutation(child_sequence, chance=0.01):
  child_genome = np.zeros(len(child_sequence), dtype=np.float32)

  # Mutation
  for a in range(len(child_sequence)):
    if random.uniform(0,1) < chance:
      child_genome[a] = random.uniform(0,1)
    else:
      child_genome[a] = child_sequence[a]

  return child_genome


# ## Create the next generation
# 
# Now, let's create a function to compute the next generation in the Genetic Algorithm (next_generation function). It performs selection, then already implemented crossover and mutation. As a result of this function, there is a new population created.
# 
# ### Selection
# Selection is a process when based on the calculated fitness function value, chromosomes to crossover are chosen. 
# 
# <img src="./assets/selection.png" alt="image" width="auto" height="400">
# 
# In this example, there is a roulette week created relative to fitness value. 
# It allows fitness proportional selection - the bigger the fitness value, the bigger the chance that a given chromosome will be selected.
# 
# The result of all the operations is returned as chromosomes.
# 

# In[ ]:


def next_generation(chromosomes, fitnesses):
  fitness_pairs = []
  fitnessTotal = 0.0
  for i in range(len(chromosomes)):
    fitness_pairs.append( [chromosomes[i], fitnesses[i]] )
    fitnessTotal += fitnesses[i]

  # Sort fitness in descending order
  fitnesses = list(reversed(sorted(fitnesses)))
  sorted_pairs = list(reversed(sorted(fitness_pairs, key=lambda x: x[1])))

  new_chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)

  # Roulette wheel
  rouletteWheel = []
  fitnessProportions = []
  for i in range(len(chromosomes)):
      fitnessProportions.append( float( fitnesses[i]/fitnessTotal ) )
      if(i == 0):
          rouletteWheel.append(fitnessProportions[i])
      else:
          rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])

  # New population
  for i in range(len(chromosomes)):

      # Selection
      spin1 = random.uniform(0, 1)
      spin2 = random.uniform(0, 1)

      j = 0
      while( rouletteWheel[j] <= spin1 ):
          j += 1

      k = 0
      while( rouletteWheel[k] <= spin2 ):
          k += 1

      parentFirst = sorted_pairs[j][0]
      parentSecond = sorted_pairs[k][0]

      # Crossover    
      child_sequence = crossover(parentFirst, parentSecond)

      # Mutation
      child_genome = mutation(child_sequence)
      
      # Add new chromosome to next population
      new_chromosomes[i] = child_genome

  return new_chromosomes


# ## Run the algorithm
# 
# Now, we can run the implemented algorithm and measure the time of the selected number of generations (set before as a 5). 
# 
# As a first population is already initialized, each generation contains the following steps:
# 
# * evaluation of the current population using eval_genomes_plain function
# * generating next generation using eval_genomes_plain function
# * wipe fitnesses values, as there is already a new generation created
# 
# The time for those operations is measured and printed after the computations. There is also the first chromosome printed to show computations were the same between both tests.

# In[ ]:


print("CPU:")
start = time.time()

# Genetic Algorithm on CPU
for i in range(num_generations):
  print("Gen " + str(i+1) + "/" + str(num_generations))
  eval_genomes_plain(chromosomes, fitnesses)
  chromosomes = next_generation(chromosomes, fitnesses) 
  fitnesses = np.zeros(pop_size, dtype=np.float32)
end = time.time()

time_cpu = end-start
print("time elapsed: " + str((time_cpu)))
print("First chromosome: " + str(chromosomes[0]))


# ## GPU execution using numba-dpex
# 
# We need to start with new population initialization, as we want to perform the same operations but now on GPU using numba-dpex implementation.
# 
# We are setting random seed the same as before to reproduce the results. 

# In[ ]:


random.seed(1111)
fitnesses = np.zeros(pop_size, dtype=np.float32)
chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
for i in range(pop_size):
  for j in range(chrom_size):
    chromosomes[i][j] = random.uniform(0,1)


# ### Evaluation function using numba-dpex
# 
# The only par that differs form the standard implementation is the evaluation function.
# 
# The most important part is to specify the index of the computation. This is the current index of the computed chromosomes. This serves as a loop function across all chromosomes.

# In[ ]:


import numba_dpex
from numba_dpex import kernel_api

@numba_dpex.kernel
def eval_genomes_sycl_kernel(item: kernel_api.Item, chromosomes, fitnesses, chrom_length):
  pos = item.get_id(0)
  num_loops = 3000
  for i in range(num_loops):
    fitnesses[pos] += chromosomes[pos*chrom_length + 1]
  for i in range(num_loops):
    fitnesses[pos] -= chromosomes[pos*chrom_length + 2]
  for i in range(num_loops):
    fitnesses[pos] += chromosomes[pos*chrom_length + 3]

  if (fitnesses[pos] < 0):
    fitnesses[pos] = 0

# Now, we can measure the time to perform some generations of the Genetic Algorithm with Data Parallel Python Numba dpex. 
# 
# Similarly like before, the time of the evaluation, creation of new generation and fitness wipe are measured for GPU execution. But first, we need to send all the chromosomes and fitnesses container to the chosen device. 

# In[ ]:


import dpnp

print("SYCL:")
start = time.time()

# Genetic Algorithm on GPU
for i in range(num_generations):
  print("Gen " + str(i+1) + "/" + str(num_generations))
  chromosomes_flat = chromosomes.flatten()
  chromosomes_flat_dpctl = dpnp.asarray(chromosomes_flat, device="gpu")
  fitnesses_dpctl = dpnp.asarray(fitnesses, device="gpu")

  exec_range = kernel_api.Range(pop_size)
  numba_dpex.call_kernel(eval_genomes_sycl_kernel, exec_range, chromosomes_flat_dpctl, fitnesses_dpctl, chrom_size)
  fitnesses = dpnp.asnumpy(fitnesses_dpctl)
  chromosomes = next_generation(chromosomes, fitnesses)
  fitnesses = np.zeros(pop_size, dtype=np.float32)


end = time.time()
time_sycl = end-start
print("time elapsed: " + str((time_sycl)))
print("First chromosome: " + str(chromosomes[0]))


# Now, let's print execution times for both CPU and GPU.

# In[ ]:


print("SYCL: ", time_sycl, " CPU: ", time_cpu)


# The time comparison is also shown in the diagram.

# In[ ]:


from matplotlib import pyplot as plt

plt.figure()
plt.title("Time comparison")
plt.bar(["Numba_dpex", "without optimization"], [time_sycl, time_cpu])

plt.show()


# # Traveling Salesman Problem
# 
# Now, let's use the knowledge about genetic algorithms to a specific problem in this code sample the Traveling Salesman Problem. There are given the cities and the distances between them. The salesman needs to visit all the cities, using possibly the shortest path. 
# 
# This problem is NP-hard and in our case, the number of possible combinations equals len(cities)! e.g. if there are 6 cities we have 720 combinations but when we have 10 cities we have over 3.000.000 combinations.
# 
# In our example we have defined:
# 
# * starting city as a 0
# * 10 cities to visit from 1 to 10
# 
# We generate distances between cities randomly in the range of defined min (100km) and max value (400km). The matrix of the distances between cities is printed after generation.

# In[ ]:


# generate distances matrix for the cities
# min length is 10 km and max length is 400km

start_city = '0'
min = 100
max = 400
cities = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

distances = np.zeros(shape=(len(cities)+1, len(cities)+1), dtype=int)

for i in range(len(cities)+1):
  for j in range(len(cities)+1):
    if i != j:
      distances[i][j] = random.randint(min-1, max+1)
    else:
      distances[i][j] = 0
  print(distances[i])


# ## Initialize population
# 
# Now, we need to initialize the population. As a chromosome, we define possible paths from city 0 to city 0 visiting all other cities.
# 
# The population size is set to 1000, but you can easily change those parameters and experiment yourself - see if the size of the population will impact the best find result. Remember, as the genetic algorithm is a heuristic it can generate different results every run.

# In[ ]:


pop_size = 1000
chrom_size = len(cities) # number of cities to visit without city the salesman is staring in
num_generations = 5

fitnesses = np.zeros(pop_size, dtype=float)
chromosomes = np.zeros(shape=(pop_size, chrom_size + 2), dtype=int)
for i in range(pop_size):
  chromosomes[i][0] = start_city # city we are starting
  to_choose = cities.copy()
  for j in range(chrom_size):
    element = random.choice(list(to_choose))
    chromosomes[i][j + 1] = element
    to_choose.remove(element) # To avoid visiting the same city twice
  chromosomes[i][chrom_size + 1] = start_city # city we are ending


# ### Evaluation function
# 
# The evaluate created generation we are calculating the full distance of the given path (chromosome). In this example, the lower the fitness value is, the better the chromosome. That's different from the general GA that we implemented.
# 
# As in this example we are also using numba-dpex, we are using an index like before.

# In[ ]:


@numba_dpex.kernel
def eval_genomes_plain_TSP_SYCL(item: kernel_api.Item, chromosomes, fitnesses, distances, pop_length):
  pos = item.get_id(1)
  for j in range(pop_length-1):
    fitnesses[pos] += distances[int(chromosomes[pos, j]), int(chromosomes[pos, j+1])]


# ### Crossover
# 
# For TSP crossover is defined in a very specific way. The first half of the child's chromosome is taken from the first parent, but the second part is in the order of the second parent. This way we can avoid broken chromosomes that don't generate any solution.

# In[ ]:


def crossover(parentFirst, parentSecond):
  child_sequence = []
  child_sequence.append(0)

  parent = parentFirst.copy()
  parent = list(filter(lambda a: a != 0, parent))
  help = parentSecond.copy()
  help = list(filter(lambda a: a != 0, help))

  for i in range(math.floor(len(parent)/2)):
    child_sequence.append(parent[i])
    help.remove(parent[i])

  child_sequence.extend(help)
  child_sequence.append(0)

  return child_sequence


# ### Mutation
# 
# For TSP the mutation we defined as a random switch of the order between 2 cities. The same as in the case of general use GA the chance of the mutation is set to 0.01. 

# In[ ]:


def mutation(chromosome, chance=0.01):
  child_genome = chromosome.copy()
  if random.uniform(0,1) < chance: # 1% chance of a random mutation
    index1 = random.randint(1, len(chromosome)-1)
    index2 = random.randint(1, len(chromosome)-1)
    if index1 != index2:
      child_genome[index1] = chromosome[index2]
      child_genome[index2] = chromosome[index1]
  return child_genome


# ### Next generation
# 
# The algorithm for generating a new population for this problem is the same - we are using a roulette wheel, but this time we need to order chromosomes in incrementing order accordingly to fitnesses. 

# In[ ]:


def next_generation_TSP(chromosomes, fitnesses):
  fitness_pairs = []
  fitnessTotal = 0.0
  for i in range(len(chromosomes)):
    fitness_pairs.append([chromosomes[i], fitnesses[i]])
    fitnessTotal += float(fitnesses[i])

  fitnesses = list(sorted(fitnesses)) #fitnesses now in order
  sorted_pairs = list(sorted(fitness_pairs, key=lambda x: x[1]))

  new_chromosomes = np.zeros(shape=(pop_size, chrom_size+2), dtype = int)

  # Create roulette wheel 
  rouletteWheel = []
  fitnessProportions = []
  for i in range(len(chromosomes)):
      fitnessProportions.append( float( fitnesses[i]/fitnessTotal ) )
      if(i == 0):
          rouletteWheel.append(fitnessProportions[i])
      else:
          rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])

  # Generate new population with children of selected chromosomes
  for i in range(len(chromosomes)):

      #Fitness Proportional Selection
      spin1 = random.uniform(0, 1)
      spin2 = random.uniform(0, 1)

      j = 0
      while( rouletteWheel[j] <= spin1 ):
          j += 1

      k = 0
      while( rouletteWheel[k] <= spin2 ):
          k += 1

      parentFirst = sorted_pairs[j][0]
      parentSecond = sorted_pairs[k][0]

      child_sequence = crossover(parentFirst, parentSecond)
      child_genome = mutation(child_sequence)

      new_chromosomes[i] = child_genome
  return new_chromosomes


# ## Algorithm execution
# 
# The execution of the algorithm looks the same, but now, we are just using the methods prepared for the Traveling Salesman Problem. 
# 
# In the end, there is the best and the worst chromosome from the last population shown together with the path distance. 

# In[ ]:


print("Traveling Salesman Problem:")

distances_dpctl = dpnp.asarray(distances, device="gpu")
# Genetic Algorithm on GPU
for i in range(num_generations):
  print("Gen " + str(i+1) + "/" + str(num_generations))
  chromosomes_flat_dpctl = dpnp.asarray(chromosomes, device="gpu")
  fitnesses_dpctl = dpnp.asarray(fitnesses.copy(), device="gpu")

  exec_range = kernel_api.Range(pop_size)
  numba_dpex.call_kernel(eval_genomes_plain_TSP_SYCL, exec_range, chromosomes_flat_dpctl, fitnesses_dpctl, distances_dpctl, pop_size)
  fitnesses = dpnp.asnumpy(fitnesses_dpctl)
  chromosomes = next_generation_TSP(chromosomes, fitnesses)
  fitnesses = np.zeros(pop_size, dtype=np.float32)

for i in range(len(chromosomes)):
  for j in range(11):
    fitnesses[i] += distances[int(chromosomes[i][j])][int(chromosomes[i][j+1])]

fitness_pairs = []

for i in range(len(chromosomes)):
  fitness_pairs.append([chromosomes[i], fitnesses[i]])

fitnesses = list(sorted(fitnesses))
sorted_pairs = list(sorted(fitness_pairs, key=lambda x: x[1]))

print("Best path: ", sorted_pairs[0][0], " distance: ", sorted_pairs[0][1])
print("Worst path: ", sorted_pairs[-1][0], " distance: ", sorted_pairs[-1][1])


# In this code sample, there was a general purpose Genetic Algorithm created and optimized using numba-dpex to run on GPU. Then the same approach was applied to the Traveling Salesman Problem.

# In[ ]:


print("[CODE_SAMPLE_COMPLETED_SUCCESFULLY]")

