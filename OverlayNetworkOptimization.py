# -*- coding: utf-8 -*-

#########################################################
#                                                       #
#       Assignment 2: GENETIC ALGORITHM APPLIED TO      #
#       OVERLAY NETWORK OPTIMIZATION                    #    
#                                                       #
# 		Author: Tyler Malmberg                          #
#                                                       #
#       Student ID: 900155614                           #
#                                                       #
#		Please DO NOT publish your implemented code     #
#       for example on GitHub							#
#                                                       #
#########################################################

import random
import numpy as np
import matplotlib.pyplot as plt


#########################################################
# PARAMETERS                                            #
#########################################################
popSize = 100
chromLength = 300
generation_max = 1000   
CROSSOVER_RATE = 0.7
MUT_RATE = 0.001 
FULL_NETWORK_COST = 30098

GENES = [1, 0]
BLANK_CHROMOSOME = np.empty([chromLength])
fitness = np.empty([popSize])                               
costVector = np.empty([chromLength])


#########################################################
# Load network                                          #
#########################################################
def loadNetwork():
    fname = "network.txt"
    input = np.loadtxt(fname)
    for i in range(0, chromLength):
        costVector[i] = input[i][2]
  

#########################################################
# FITNESS EVALUATION                                    #
#########################################################         
def evaluateBestFitness(chromosome, best):
    evalFit = chromosome.getFitness()

    if evalFit >= best:
        best = evalFit        
    return best


#########################################################
# PERFORMANCE GRAPH                                     #
#########################################################
def plotChart(best,avg):
    plt.plot(best,label='best')
    plt.plot(avg,label='average')
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    plt.legend()
    plt.xlim(1,generation_max-1)    
    plt.ylim(0.0, 1.0)
    plt.show()


#########################################################
# INDIVIDUAL CLASS                                      #
# Represents an individual in a population              #             
#########################################################
class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calc_fitness()

    def getFitness(self):
        return self.fitness

    def getChromosome(self):
        return self.chromosome

    def setChromosome(self, chrome):
        self.chromosome = chrome

    def getGene(self, index):
        return self.chromosome[index]

    @classmethod
    def mutate_gene(cls):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_chrom(cls):
        global chromLength
        global BLANK_CHROMOSOME
        chromosome = BLANK_CHROMOSOME
        for i in range(0, chromLength):
            chromosome[i] = int(cls.mutate_gene())
        return chromosome

    # TODO
    def mate(self, partner):
        chromeLen = len(self)
        child1_chrome = []
        child2_chrome = []
        crossPt1 = random.randint(0, chromeLen - 1)
        #crossPt2 = random.randint(0, chromeLen - 1)

        child1_chrome = np.hstack((self.chromosome[0:crossPt1], partner.chromosome[crossPt1:]))
        child2_chrome = np.hstack((partner.chromosome[0:crossPt1], self.chromosome[crossPt1:]))

        child1 = Individual(child1_chrome)
        child2 = Individual(child2_chrome)

        return child1, child2

    def calc_fitness(self):
        global FULL_NETWORK_COST
        global costVector
        costFullyConnectedNetwork = FULL_NETWORK_COST
        global chromLength
        length = chromLength
        cost = 0
        fit = 0

        for i in range(0, length):        
            if self.getGene(i) == 1:
                cost = cost + costVector[i]
        fit = 1 - (cost / costFullyConnectedNetwork)
    

        return fit
        

#########################################################
# MAIN                                                  #
#########################################################
def main():
    global popSize
    best = 0.0
    average = 0.0
    generation = 0
    fitness_total = 0.0
    fitness_average = 0.0
    loadNetwork()
    bestM = np.empty([generation_max], dtype = np.float32)
    averageM = np.empty([generation_max], dtype = np.float32)
    print("GENETIC ALGORITHM APPLIED TO OVERLAY NETWORK OPTIMIZATION")

    #Initialize Population
    population = []
    for i in range(0, popSize):
        chrome = Individual.create_chrom()
        population.append(Individual(chrome))

    # VVV Initial plot values
    bestM[generation] = best
    averageM[generation] = average
    while (generation < generation_max):
        # RUNNING GA
        # ... to be implemented
        average = 0
        best = 0
        for i in range(0, popSize):
            num = evaluateBestFitness(population[i], best)
            if num > best:
                best = num
        bestM[generation] = best
        
        fitness_total = 0
        for i in range(0, popSize):  
            currentFitness = population[i].getFitness()
            fitness_total = fitness_total + currentFitness
        average = fitness_total/popSize

        print("Generation: ", generation, " Average fitness: ", average)
        #print("Generation: ", generation, " Best fitness: ", best)
        averageM[generation] = average
        generation = generation + 1

    print("best fitness: ", best)   
    
    
    plotChart(bestM, averageM)

#########################################################
# RUN MAIN                                              #
#########################################################

if __name__ == '__main__':
    main()
