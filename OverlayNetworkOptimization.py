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
# TOURNAMENT METHOD                                     #
#########################################################

def tournament(fighter1, fighter2):
    if fighter1.getFitness() > fighter2.getFitness():
        winner = fighter1
    elif fighter1.getFitness() < fighter2.getFitness():
        winner = fighter2
    else:
        winner = fighter1

    return winner

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
        return [cls.mutate_gene() for _ in range(chromLength)]

    def mate(self, partner):
        global chromLength
        chromeLen = chromLength
        prob = random.randint(0, 100)

        if prob <= 70:
            child1_chrome = []
            child2_chrome = []
            crossPt1 = random.randint(0, chromeLen - 1)
            crossPt2 = random.randint(0, chromeLen - 1)

            child1_chrome = np.hstack((self.chromosome[0:crossPt1], partner.chromosome[crossPt1:]))
            child2_chrome = np.hstack((partner.chromosome[0:crossPt1], self.chromosome[crossPt1:]))

            child1_chrome = np.hstack((child1_chrome[0:crossPt2], child2_chrome[crossPt2:]))
            child2_chrome = np.hstack((child2_chrome[0:crossPt2], child1_chrome[crossPt2:]))

            child1 = Individual(child1_chrome)
            child2 = Individual(child2_chrome)
            
            # if prob <= 1:
            #     child1_chrome = Individual.create_chrom
            #     child1
            return child1, child2
        else:
            return self, partner


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

        # Calculate Best Fitness
        for i in range(0, popSize):
            num = evaluateBestFitness(population[i], best)
            if num > best:
                best = num
        bestM[generation] = best
        
        # Calculate Avg Fitness
        fitness_total = 0
        currentFitness = 0
        for i in range(0, popSize):  
            currentFitness = population[i].getFitness()
            fitness_total = fitness_total + currentFitness
        average = fitness_total/popSize

        print("Generation: ", generation, " Average fitness: ", average)
        #print("Generation: ", generation, " Best fitness: ", best)
        averageM[generation] = average

        # Create New Population
        new_population = []
        while (len(new_population) <= popSize):
            fighter1 = population[random.randint(0, popSize - 1)]
            fighter2 = population[random.randint(0, popSize - 1)]

            mate1 = tournament(fighter1, fighter2)

            fighter1 = population[random.randint(0, popSize - 1)]
            fighter2 = population[random.randint(0, popSize - 1)]

            mate2 = tournament(fighter1, fighter2)

            child1, child2 = mate1.mate(mate2)

            new_population.extend([child1, child2])

        population = new_population
        generation = generation + 1

    print("best fitness: ", best)   
    
    
    plotChart(bestM, averageM)


#########################################################
# TESTING METHOD                                        #
#########################################################

def test():
    chrome = Individual.create_chrom()
    chrome2 = Individual.create_chrom()

    peep1 = Individual(chrome)
    peep2 = Individual(chrome2)

    print(chrome)
    print(chrome2)

#########################################################
# RUN MAIN                                              #
#########################################################

if __name__ == '__main__':
    main()
    #test()
