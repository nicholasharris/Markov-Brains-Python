##################################
# A demonstration of how to use this python implmentation of Markov Network Brains
# See "markov.py" for the full implementation code.
#
# This template should allow you to easily introduce any problem domain and use this implementation with it.
#
# Nicholas Harris
##################################

import random
import markov


# This evaluation function contains, in essence, your problem domain. Within it,
#   you should evaluate every member of the population and assign it fitness >= 0 (depending
#   on its performance in the problem domain)
def evaluate_individual(brain):

    #You have free reign here. To use the individual brain you provide inputs as a list, and may activate the brain
    # as many times as you wish. For example:

    myInputs = [ 1, 0, 1, 1 ]   #Inputs can be taken from an artifial environment, for example
    brain.activate(myInputs)  #Inputs are automatically added starting from the left of the brain's brain state

    #activate may be called many times as inputs are updated from environment, for example

    output = brain.brainState[len(brain.brainState) - 1]  #Any value from the brain state can be considered as an output; the last
                                                        # value seems a natural choice as it is far away from the inputs
                
    #This output can be considered as an action, such as movement


    #When whatever trial you define has been concluded, return a fitness based on the individual's performance
    myFitness = 100.0
    return myFitness


#This function lets you easily run the GA. Just call this function repeatedly to create successive generations
#  that (hopefully) increasingly improve on the problem domain
def Eval_Genomes(myPopulation):  

        #First you must assign a fitness to every member of the population, perhaps like so:
        for x in range(len(myPopulation.brains)):
            myPopulation.brains[x].fitness = evaluate_individual(myPopulation.brains[x]) #I imagine you define your problem domain in
                                                                                            # a separate function called evaluate_individual
                                                                                            # which accepts a markov brain as an argument
                                                                                            # And returns a fitness ( >= 0) for it 

        myPopulation.eval_genomes() #Call this after all fitnesses have been assigned
                                    #  to automatically create a new population based on fitnesses and
                                    #  other Genetic Algorithm parameters

        
random.seed()

#CREATE THE MARKOV POPULATION, which is the highest level object we define in this implementation
# The parameters are listed below, followed by a more detailed description of each.
# PARAMETERS: (popSize, brainSize, genome_length, brain_steps, elitism, diversity_generate, point_mutation, insert_mutation, delete_mutation,
#           copy_mutation, big_delete_mutation, big_copy_mutation, prob_engulf, fitness_sharing (bool), preserve_diversity (bool))
#
# popSize: size of the population (int)
# brainSize: the size of the brain state of each Markov Brain (int)
# genome length: The maximum length of each genome in the population (int). Bigger genome create more complex networks
# brain_steps: The number of times the network is fired in each activation (int)
# elitism: the number of brains preserved as-is into the next generation (int)
# diversity_generate: The number of brain in the "next" generation produced randomly, rather than via crossover-reproduction (int)
# point_mutation: The probability of a given value in the genome during reproduction (float 0 to 1)
# insert_mutation: The probabiliity of an insert mutation in the genome during reproduction (float 0 to 1)
# delete_mutation: The probabiliity of a given value being deleted from the genome during reproduction (float 0 to 1)
# copy_mutation: The probabiliity of a given value being copied in the genome during reproduction (float 0 to 1)
# big_delete_mutation: The probabiliity of a big chunk of values being deleted in the genome during reproduction (float 0 to 1)
# big_copy_mutation: The probabiliity of a big chunk of values being copied in the genome during reproduction (float 0 to 1)
# prob_engulf: The probability of one genome "engulfing" another during reproduction (that is, splicing an entire genome into another, rather than ordinary corssover). (float 0 to 1)
# fitness_sharing: GA parameter to reduce the fitness of individuals who have the same fitness (i.e. they fill the same niche, it is thought.) (bool).
# preserve_diversity: When true, only one copy of a brain with the same fitness can be preserved through elitism (bool)
myPopulation = markov.MarkovPopulation(500, 64, 5000, 1, 30, 70, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0, True, True)


# Run for an arbitary number of generations, or set your own stopping point (based on highest attained fitness perhaps)
NUM_GENERATIONS = 5000
count = 0
print("Running Eval Genomes Loop")
while(count < NUM_GENERATIONS):
    Eval_Genomes(myPopulation)
    count += 1


