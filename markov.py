##################################
# A (probably flawed) implementation of Markov brains,
# Based on article: http://devosoft.org/a-quick-introduction-to-markov-network-brains/
#   from devolab of Michigan State University
#
# Nicholas Harris
##################################

##################################
# In this code the greatest emphasis was placed on clarity of structure and readability.
# No effort has been made to optimize.
#
# See "example.py" for a demonstration on using the code defined here.
##################################
import random
import math
import numpy
from functools import reduce
from collections import Counter


#These constants are arbitrary and may be changed based on intuition, performance constraints,
#  or any other reason.
global THRESHOLD_CONSTANT
global TIMER_CONSTANT
THRESHOLD_CONSTANT = 48
TIMER_CONSTANT = 96
MAX_GENOME_CONSTANT = 50000

#Function to produce ternary string of decimal number (used for ternary gate)
def ternary(n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

class MarkovPopulation:
    def __init__(self, popSize, brainSize, genome_length, brain_steps, elitism, diversity_generate, point_mutation, insert_mutation, delete_mutation, copy_mutation, big_delete_mutation, big_copy_mutation, prob_engulf, fitness_sharing, preserve_diversity):
        self.brains = []
        self.idCounter = 0
        self.elitism = elitism
        self.diversity_generate = diversity_generate
        self.popSize = popSize
        self.brainSize = brainSize
        self.brain_steps = brain_steps
        self.genome_length = genome_length
        self.point_mutation = point_mutation
        self.insert_mutation = insert_mutation
        self.delete_mutation = delete_mutation
        self.copy_mutation = copy_mutation
        self.big_delete_mutation = big_delete_mutation
        self.big_copy_mutation = big_copy_mutation
        self.prob_engulf = prob_engulf
        self.gen = 0
        self.fitness_sharing = fitness_sharing
        self.preserve_diversity = preserve_diversity
        
        for x in range(popSize):
            self.brains.append(MarkovBrain(self.brainSize, self.genome_length, self.brain_steps, self.idCounter))
            self.idCounter = self.idCounter + 1

    def eval_genomes(self):
        print("\n\n****** MARKOV PYTHON ||||| GENERATION STEP " + str(self.gen) + " ********\n")
        fitnesses = []
        fitnessTotal = 0.0
        fitness_pairs = []
        for x in range(self.popSize):
            fitnesses.append(self.brains[x].fitness)
            fitnessTotal += self.brains[x].fitness
            fitness_pairs.append( [self.brains[x], fitnesses[x]] )


        normalized_fitnesses = []
        normalized_fitness_pairs = []
        normalized_fitness_total = 0
        if self.fitness_sharing == True:    #If fitness sharing is on, normalize the genome's fitnesses
            counted_fitnesses = Counter(fitnesses)
            for x in range(self.popSize):
                normalized_fitnesses.append( fitnesses[x]/counted_fitnesses[fitnesses[x]] )
                normalized_fitness_pairs.append( [self.brains[x], normalized_fitnesses[x] ])
                normalized_fitness_total += normalized_fitnesses[x]
                self.brains[x].normalized_fitness = normalized_fitnesses[x]

        fitnesses = list(reversed(sorted(fitnesses))) #fitnesses now in descending order
        sorted_pairs = list(reversed(sorted(fitness_pairs, key=lambda x: x[1])))

        

        new_brains = []
        new_brains_fitnesses = []
        #preserve some brains through elitism
        if self.preserve_diversity == False:
            for x in range(self.elitism):
                new_brains.append(sorted_pairs[x][0])
        else:
            count = 0
            while len(new_brains) < self.elitism and count < self.popSize:
                if fitnesses[count] not in set(new_brains_fitnesses):
                    new_brains.append(sorted_pairs[count][0])
                    new_brains_fitnesses.append(fitnesses[count])
                count += 1
                

        best_brain = sorted_pairs[0][0]
        

        if (self.fitness_sharing == False):
            print("   Best brain ID: " + str(best_brain.ID) + "  ||  Fitness: " + str(best_brain.fitness) + "\n  ||  # Gates: " + str(len(best_brain.gates)) + "  ||  Genome Length: " + str(best_brain.genome.length) + "\n")
        else:
            print("   Best brain ID: " + str(best_brain.ID) + "  ||  Fitness: " + str(best_brain.fitness) + "  ||  Normalized Fitness: " + str(best_brain.normalized_fitness) + "\n  ||  # Gates: " + str(len(best_brain.gates)) + "  ||  Genome Length: " + str(best_brain.genome.length) + "\n")
        gatestring = []
        for gate in best_brain.gates:
            gatestring.append(gate.type)
        gatestring = str(gatestring)
        print("   Gate Types: " + gatestring + "\n")
        average_fitness = reduce(lambda x, y: x + y, fitnesses)/float(len(fitnesses))
        print("   Average Fitness: " + str(average_fitness) + "\n")

        #If using fitness sharing, replace the notion of fitness now with the normalized fitness, after elitism has been preserved.
        if (self.fitness_sharing == True):
            fitnesses = list(reversed(sorted(normalized_fitnesses)))
            sorted_pairs = list(reversed(sorted(normalized_fitness_pairs, key=lambda x: x[1])))
            fitnessTotal = normalized_fitness_total

        #create roulette wheel from relative fitnesses for fitness proportional selection
        rouletteWheel = []
        fitnessProportions = []
        for i in range(self.popSize):
            fitnessProportions.append( float( fitnesses[i]/fitnessTotal ) )
            if(i == 0):
                rouletteWheel.append(fitnessProportions[i])
            else:
                rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])
    
        #Generate most new population with children of selected brains
        while len(new_brains) < self.popSize - self.diversity_generate:

            #Fitness Proportional Selection
            spin1 = random.uniform(0, 1)      # A random float from 0.0 to 1.0
            spin2 = random.uniform(0, 1)      # A random float from 0.0 to 1.0

            j = 0
            while( rouletteWheel[j] <= spin1 ):
                j += 1

            k = 0
            while( rouletteWheel[k] <= spin2 ):
                k += 1

                
            genome_copy = Genome(sorted_pairs[j][0].genome.length)    #Genome of parent 1
            genome_copy2 = Genome(sorted_pairs[k][0].genome.length)  #Genome of parent 2
            sequence_copy = []
            sequence_copy2 = []
            for value in sorted_pairs[j][0].genome.sequence:
                sequence_copy.append(value)
            for value in sorted_pairs[k][0].genome.sequence:
                sequence_copy2.append(value)
            genome_copy.sequence = sequence_copy
            genome_copy2.sequence = sequence_copy2

            #create child genome from parents'
            index = random.randint(0, genome_copy.length - 1)
            index2 = random.randint(0, genome_copy2.length - 1)

            child_sequence = []

            for y in range(math.floor(genome_copy.length / 2)):
                child_sequence.append( genome_copy.sequence[ (index + y) % genome_copy.length ] )

            for y in range(math.floor(genome_copy2.length / 2)):
                child_sequence.append( genome_copy2.sequence[ (index2 + y) % genome_copy2.length ] )


            child_genome = Genome(1) #temporarily initialized

            e_spin = random.uniform(0, 1)
            if e_spin < self.prob_engulf:   #If genome was engulfed, do this instead of crossover
                child_genome = Genome( genome_copy.length + genome_copy2.length )
                engulf_point = random.randint(0, genome_copy.length - 1)
                for w in range(engulf_point):
                    child_genome.sequence[w] = genome_copy.sequence[w]
                for v in range(genome_copy2.length):
                    child_genome.sequence[engulf_point + v] = genome_copy2.sequence[v]
                for u in range(genome_copy.length - engulf_point):
                    child_genome.sequence[engulf_point + genome_copy2.length + u] = genome_copy.sequence[engulf_point + u]
            else:   #Use child obtained from normal crossover 
                child_genome = Genome( len(child_sequence) )

                for y in range(len(child_genome.sequence)):
                    child_genome.sequence[y] = child_sequence[y] 

            #mutate genome
            child_genome.mutate(self.point_mutation, self.insert_mutation, self.delete_mutation, self.copy_mutation, self.big_delete_mutation, self.big_copy_mutation)

            #Add brain based on new genome to new popultaion
            new_brains.append(MarkovBrain(self.brainSize, child_genome.length, self.brain_steps, self.idCounter, child_genome))
            self.idCounter = self.idCounter + 1
        #Generate a number of random individuals for diversity in the population
        for x in range(self.diversity_generate):
            new = MarkovBrain(self.brainSize, self.genome_length, self.brain_steps, self.idCounter)
            new_brains.append(new)
            self.idCounter = self.idCounter + 1

        self.brains = new_brains
        random.shuffle(self.brains)
        self.gen = self.gen + 1
        print("\n*************** GENERATION STEP FINISH ********************\n\n")

class MarkovBrain:
    def __init__(self, size, genome_length, brain_steps, ID, genome_ready = None):
        self.ID = ID
        self.size = size
        if genome_ready is None:
            self.genome = Genome(genome_length)
        else:
            self.genome = Genome(genome_length)
            for x in range(len(self.genome.sequence)):
                self.genome.sequence[x] = genome_ready.sequence[x]
        self.brainState = []
        self.newBrainState = []
        self.brainFlags = []
        self.brain_steps = brain_steps
        self.fitness = 0.0
        self.validation_fitness = 0.0
        self.normalized_fitness = 0.0
        
        for x in range(size):
            self.brainState.append(0)
            self.brainFlags.append(0)
            self.newBrainState.append(0)
        self.gates = []
        for y in range(genome_length):
            #deterministic logic gate
            if self.genome.sequence[y] == 42 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 213:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                tableValues = []
                for x in range(2**numInputs):
                    bString = ('{0:04b}'.format(self.genome.sequence[(y + 12 + x) % len(self.genome.sequence)]))

                    value = []
                    for z in range(numOutputs):
                        value.append(int(bString[len(bString) - 1 - z]))

                    tableValues.append(value)
                
                self.gates.append(DeterministicGate( numInputs, numOutputs, inputIndices, outputIndices, tableValues))
            #probabilistic logic gate
            elif self.genome.sequence[y] == 43 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 212:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                tableValues = []
                for x in range(2**numInputs):
                    row = []
                    for z in range(2**numOutputs):
                        row.append(self.genome.sequence[(y + 12 + x) % len(self.genome.sequence)])
                    tableValues.append(row)
                #Normalize table values so that probabilities in each row sum to one
                for x in range(2**numInputs):
                    denom = 0
                    for z in range(2**numOutputs):
                        denom += (1 + tableValues[x][z])
                    for z in range(2**numOutputs):
                        tableValues[x][z] = (tableValues[x][z] + 1)/denom
                      
                self.gates.append(ProbabilisticGate( numInputs, numOutputs, inputIndices, outputIndices, tableValues))
            #threshold gate
            elif self.genome.sequence[y] == 44 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 211:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                
                threshold = (self.genome.sequence[(y + 12) % len(self.genome.sequence)]) % THRESHOLD_CONSTANT
                
                self.gates.append(ThresholdGate( numInputs, numOutputs, inputIndices, outputIndices, threshold))
            #timer gate
            elif self.genome.sequence[y] == 45 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 210:  
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

        
                timer = (self.genome.sequence[(y + 12) % len(self.genome.sequence)]) % TIMER_CONSTANT
                
                self.gates.append(TimerGate(numOutputs, outputIndices, timer))
            #ternary logic gate
            elif self.genome.sequence[y] == 46 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 209:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                tableValues = []
                for x in range(3**numInputs):
                    tString = ternary(self.genome.sequence[(y + 12 + x) % len(self.genome.sequence)])

                    tString = list(tString)

                    while len(tString) < numOutputs:
                        tString = [0] + tString
                    

                    value = []
                    #change ternary values from (0, 2) range to (-1, 1) range
                    for z in range(numOutputs):
                        value.append(int(tString[len(tString) - 1 - z]) - 1)

                    
                    tableValues.append(value)
                
                self.gates.append(TernaryGate( numInputs, numOutputs, inputIndices, outputIndices, tableValues))
            #Artificial Neuron gate
            elif self.genome.sequence[y] == 47 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 208:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                numNeurons = (self.genome.sequence[(y + 12) % len(self.genome.sequence)] % 4) + 1

                weights = []
                for x in range((numInputs + 1) * (numNeurons)):
                    weights.append( (self.genome.sequence[(y + 13 + x) % len(self.genome.sequence)] - 128.0)/100.0 )

                
                self.gates.append(NNGate( numInputs, numOutputs, inputIndices, outputIndices, numNeurons, weights))
            #Stats Gate (min max or average)
            elif self.genome.sequence[y] == 48 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 207:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                operation = self.genome.sequence[(y + 12) % len(self.genome.sequence)] % 3
                
                self.gates.append(StatsGate( numInputs, numOutputs, inputIndices, outputIndices, operation))
            
            #Sum Gate
            elif self.genome.sequence[y] == 49 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 206:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)
                
                self.gates.append(SumGate( numInputs, numOutputs, inputIndices, outputIndices))
            
            #NULL gate
            elif self.genome.sequence[y] == 50 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 205:  
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
               
                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)
                
                self.gates.append(NullGate(numOutputs, outputIndices))
            
            #Invert Gate
            elif self.genome.sequence[y] == 51 and self.genome.sequence[(y + 1) % len(self.genome.sequence)] == 204:  
                numInputs = (self.genome.sequence[(y + 2) % len(self.genome.sequence)] % 4) + 1
                numOutputs = (self.genome.sequence[(y + 3) % len(self.genome.sequence)] % 4) + 1
                inputIndices = []
                for x in range(numInputs):
                    inputIndices.append(self.genome.sequence[(y + 4 + x) % len(self.genome.sequence)] % self.size)

                outputIndices = []
                for x in range(numOutputs):
                    outputIndices.append(self.genome.sequence[(y + 8 + x) % len(self.genome.sequence)] % self.size)

                operation = self.genome.sequence[(y + 12) % len(self.genome.sequence)] % 2
                
                self.gates.append(InvertGate( numInputs, numOutputs, inputIndices, outputIndices, operation))
            

                
    def activate(self, inputs):
        for a in range(self.brain_steps):   # steps of thinking per action
            for value in self.brainFlags:
                value = 0
            #overwrite appropriate values in brain state with inputs
            for x in range(len(inputs)):
                self.brainState[x] = inputs[x]

            self.newBrainState = self.brainState

            if self.gates is None:
                for x in range(len(self.brainState)):
                    self.brainState[x] = 0
                    self.newBrainState[x] = 0
                
            else:
                for gate in self.gates:
                    gate.activate(self.newBrainState, self.brainState, self.brainFlags)

            self.brainState = self.newBrainState
     
class Genome:
    def __init__(self, length):
        self.length = length
        self.sequence = []
        for x in range(self.length):
            self.sequence.append(random.randint(0, 255))

    def mutate(self, point_mutation, insert_mutation, delete_mutation, copy_mutation, big_delete_mutation, big_copy_mutation):
        #big delete mutation (1000 min length of genome necessary)
        if len(self.sequence) >= 1000 and random.uniform(0, 1) <= big_delete_mutation:
            delete_length = random.randint(256, 512)
            delete_index = random.randint(0, len(self.sequence) - 1)

            for x in range(delete_length):
                self.sequence[delete_index] = -1
                delete_index = (delete_index + 1) % len(self.sequence)

            obj = -1
            for x in range(delete_length):
                self.sequence.remove(obj)
            
        #big copy mutation(20,000 max length of genome)
        if len(self.sequence) < MAX_GENOME_CONSTANT and random.uniform(0, 1) <= big_copy_mutation:
            copy_length = random.randint(256, 512)
            copy_index = random.randint(0, len(self.sequence) - 1)
            copied_genes = []
            for x in range(copy_length):
                copied_genes.append(self.sequence[copy_index])
                copy_index = (copy_index + 1) % len(self.sequence)

            for x in range(copy_length):
                self.sequence.insert(copy_index, copied_genes[x])
                copy_index = (copy_index + 1) % len(self.sequence)
                
        #all single-point mutations
        for y in range(len(self.sequence)):
            #point mutation
            if random.uniform(0, 1) <= point_mutation and y < len(self.sequence):
                self.sequence[y] = random.randint(0, 255)
            #insert mutation
            if len(self.sequence) < MAX_GENOME_CONSTANT and random.uniform(0, 1) <= insert_mutation and y < len(self.sequence):
                self.sequence.insert(y, random.randint(0, 255))
            #delete mutation
            if len(self.sequence) > 1000 and random.uniform(0, 1) <= delete_mutation and y < len(self.sequence):
                obj = -1
                self.sequence[y] = -1
                self.sequence.remove(obj)
            #copy mutation
            if len(self.sequence) < MAX_GENOME_CONSTANT and random.uniform(0, 1) <= copy_mutation and y < len(self.sequence):
                self.sequence.insert(y, self.sequence[y])

        self.length = len(self.sequence)

class DeterministicGate:
    def __init__(self, numInputs, numOutputs, inputIndices, outputIndices, tableValues ):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.tableValues = tableValues
        self.type = "D"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])

        #discretize the inputs
        for x in range(len(inputs)):
            if inputs[x] > 0:
                inputs[x] = 1
            else:
                inputs[x] = 0

        tableIndex = 0
        inputs = list(reversed(inputs))
        for x in range(len(inputs)):
            tableIndex = tableIndex + (inputs[x] * (2**x))

        output = self.tableValues[tableIndex]

        #Overwriting outputs are summed (used to be OR'd)
        for x in range(self.numOutputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output[x]
                brainFlags[self.outputIndices[x]] = 1
            else:
                newBrainState[self.outputIndices[x]] = output[x] + newBrainState[self.outputIndices[x]]

class NNGate:
    def __init__(self, numInputs, numOutputs, inputIndices, outputIndices, numNeurons, weights ):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.numNeurons = numNeurons
        self.weights = weights
        self.type = "N"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])

        output = []
        for x in range(self.numOutputs):
            output.append(0)
            
        for x in range(self.numNeurons):
            activation = 0
            for y in range(len(inputs)):
                activation += inputs[y] * self.weights[y]
            activation += self.weights[len(self.weights) - 1]  #bias term, last of weight parameters

            activation = math.tanh(activation)
            
            for y in range(self.numOutputs):
                output[y] += activation

        #Overwriting outputs are summed (used to be OR'd)
        for x in range(self.numOutputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output[x]
                brainFlags[self.outputIndices[x]] = 1
            else:
                newBrainState[self.outputIndices[x]] = output[x] + newBrainState[self.outputIndices[x]]
                

class ProbabilisticGate:
    def __init__(self, numInputs, numOutputs, inputIndices, outputIndices, tableValues ):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.tableValues = tableValues
        self.type = "P"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])

        #discretize the inputs
        for x in range(len(inputs)):
            if inputs[x] > 0:
                inputs[x] = 1
            else:
                inputs[x] = 0

        tableIndex = 0
        inputs = list(reversed(inputs))
        for x in range(len(inputs)):
            tableIndex = int(tableIndex + (inputs[x] * (2**x)))

        
        roll = random.uniform(0, 1)
        outputNum = 0
        for x in range(len(self.tableValues[tableIndex])):
            probSum = 0
            for y in range(x):
                probSum += self.tableValues[tableIndex][y]
            probSum += self.tableValues[tableIndex][x]
            if roll < probSum:
                outputNum = x
                break
        output = [int(x) for x in list('{0:04b}'.format(outputNum))]
        for x in range(self.numOutputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output[x]
                brainFlags[self.outputIndices[x]] = 1
            else:
               newBrainState[self.outputIndices[x]] = output[x] + newBrainState[self.outputIndices[x]]
                

class ThresholdGate:
    def __init__(self, numInputs, numOutputs, inputIndices, outputIndices, threshold ):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.accumulator = 0
        self.threshold = threshold
        self.type = "Th"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])

        #discretize the inputs
        for x in range(len(inputs)):
            if inputs[x] > 0:
                inputs[x] = 1
            else:
                inputs[x] = 0

        incoming = 0
        for x in range(len(inputs)):
            if inputs[x] == 1:
                incoming = incoming + 1

        self.accumulator = self.accumulator + incoming

        #Fire gate (return 1 on all outputs) if threshold is reached
        if self.accumulator > self.threshold:
            for x in range(self.numOutputs):
                if brainFlags[self.outputIndices[x]] == 0:
                    newBrainState[self.outputIndices[x]] = 1
                    brainFlags[self.outputIndices[x]] = 1
                else:
                    newBrainState[self.outputIndices[x]] += 1
                    #Else do nothing (outputs are OR'd)

            #Reset the accumulator to 0 after firing
            self.accumulator = 0
            
class TimerGate:
    def __init__(self, numOutputs, outputIndices, timer ):
        self.numOutputs = numOutputs
        self.outputIndices = outputIndices
        self.updates = 0
        self.timer = timer
        self.type = "T"

    def activate(self, newBrainState, brainState, brainFlags):
        self.updates += 1

        #Fire gate (return 1 on all outputs) if timer is surpassed
        if self.updates > self.timer:
            for x in range(self.numOutputs):
                if brainFlags[self.outputIndices[x]] == 0:
                    newBrainState[self.outputIndices[x]] = 1
                    brainFlags[self.outputIndices[x]] = 1
                else:
                    if newBrainState[self.outputIndices[x]] == 0:
                        newBrainState[self.outputIndices[x]] = 1
                    #Else do nothing (outputs are OR'd)
            self.updates = 0

class TernaryGate:
    def __init__(self, numInputs, numOutputs, inputIndices, outputIndices, tableValues ):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.tableValues = tableValues
        self.type = "Ter"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])

        #discretize the inputs (differently from before as inputs must be ternary)
        for x in range(len(inputs)):
            if inputs[x] >= 1:
                inputs[x] = 1
            elif inputs[x] <= -1:
                inputs[x] = -1
            else:
                inputs[x] = 0

        tableIndex = 0
        inputs = list(reversed(inputs))
        for x in range(len(inputs)):
            tableIndex = int(tableIndex + ((inputs[x] + 1) * (3**x)))

        output = self.tableValues[tableIndex]

        #Overwriting outputs are summed (used to be OR'd)
        for x in range(self.numOutputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output[x]
                brainFlags[self.outputIndices[x]] = 1
            else:
                newBrainState[self.outputIndices[x]] = output[x] + newBrainState[self.outputIndices[x]]       

     
class StatsGate:
    def __init__ (self, numInputs, numOutputs, inputIndices, outputIndices, operation):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.operation = operation
        self.type = "S"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])

        if self.operation == 0:  #min
            output = min(inputs)
        elif self.operation == 1: #max
            output = max(inputs)
        elif self.operation == 2: #avg
            output = sum(inputs)/(len(inputs))
            

        #Overwriting outputs are summed (used to be OR'd)
        for x in range(self.numOutputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output
                brainFlags[self.outputIndices[x]] = 1
            else:
                newBrainState[self.outputIndices[x]] = output + newBrainState[self.outputIndices[x]]    


class SumGate:
    def __init__ (self, numInputs, numOutputs, inputIndices, outputIndices ):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.type = "Sum"

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])
        
        output = sum(inputs)
            
        #Overwriting outputs are summed (used to be OR'd)
        for x in range(self.numOutputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output
                brainFlags[self.outputIndices[x]] = 1
            else:
                newBrainState[self.outputIndices[x]] = output + newBrainState[self.outputIndices[x]] 

class NullGate:
    def __init__ (self, numOutputs, outputIndices):
        self.numOutputs = numOutputs
        self.outputIndices = outputIndices
        self.type = "Nu"

    def activate(self, newBrainState, brainState, brainFlags):           
        #Overwriting outputs 
        for x in range(self.numOutputs):
            newBrainState[self.outputIndices[x]] = 0
            brainFlags[self.outputIndices[x]] = 0


class InvertGate:
    def __init__ (self, numInputs, numOutputs, inputIndices, outputIndices, operation):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.inputIndices = inputIndices
        self.outputIndices = outputIndices
        self.operation = operation
        self.type = "I"

        index = len(self.outputIndices)
        while (len(self.outputIndices) < len(self.inputIndices)):
            self.outputIndices.append(self.inputIndices[index])
            index += 1

    def activate(self, newBrainState, brainState, brainFlags):
        inputs = []
        for x in range(self.numInputs):
            inputs.append(brainState[self.inputIndices[x]])
        output = []
        for x in range(self.numInputs):
            if self.operation == 0:  #negative
                output.append(inputs[x] * -1.0)
            elif self.operation == 1: #inverse
                if inputs[x] > 0.001 or inputs[x] < -0.001:
                    output.append(1.0/inputs[x])
                else:
                    output.append(1.0)           

        #Overwriting outputs are summed (used to be OR'd)
        for x in range(self.numInputs):
            if brainFlags[self.outputIndices[x]] == 0:
                newBrainState[self.outputIndices[x]] = output[x]
                brainFlags[self.outputIndices[x]] = 1
            else:
                newBrainState[self.outputIndices[x]] = output[x] + newBrainState[self.outputIndices[x]]    
