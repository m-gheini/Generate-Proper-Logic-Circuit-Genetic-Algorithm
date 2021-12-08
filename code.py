import random
import csv
import heapq
import time

possibleGenes = ["AND", "NAND", "OR", "NOR", "XOR", "XNOR"]
population = 100

class Chromosome(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculateFitness()

    @classmethod
    def getRandomGene(self):
        return random.choice(possibleGenes)

    @classmethod
    def createRandom(self, targetLen):
        result = []
        for i in range(targetLen):
            result.append(self.getRandomGene())
        return result

    def calculateFitness(self):
        t = Test()
        fitness = 0
        for row in Test.rows[1:]:
            res = t.getResultedOutput(row, self.chromosome)
            actualRes = t.getWantedOutput(row)
            if(res == actualRes):
                fitness += 1
        return fitness

    def __lt__(self, other):
        return self.fitness > other.fitness

    def crossover(self, other, targetLen):
        probability = random.uniform(0, 1)
        if probability <= 0.25:
            return (self, other)
        else:
            point1 = int(random.uniform(0, targetLen/2))
            point2 = int(random.uniform((targetLen/2)+1, targetLen-2))
            new1 = self.chromosome[0:point1] + other.chromosome[point1:point2] + self.chromosome[point2:]
            new2 = other.chromosome[0:point1] + self.chromosome[point1:point2] + other.chromosome[point2:]
            return (Chromosome(new1), Chromosome(new2))

    def mutation(self, size, stuck):
        if not stuck:
            probability = 1 - (self.fitness/size)
        if stuck:
            probability = (1 - (self.fitness/size)) + 0.1
        newChromosome = []
        for gene in self.chromosome:
            chance = random.uniform(0, 1)
            if chance <= probability:
                gene = self.getRandomGene()
            newChromosome.append(gene)
        return Chromosome(newChromosome)

class Generation():
    sortedGeneration = []
    def __init__(self):
        self.generation = []

    def add(self, chromosome):
        heapq.heappush(self.generation, chromosome)

    def empty(self):
        return len(self.generation) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            chromosome = heapq.heappop(self.generation)
            return chromosome

    def sort(self):
        while(not(self.empty())):
            Generation.sortedGeneration.append(self.remove())

    def calculateProbability(self):
        probability = []
        for i in range(len(Generation.sortedGeneration)):
            if i < int(0.25 * population):
                probability.append(4)
            ## For new selection method
            # if i < int(0.25 * population):
            #     continue
            elif i < int(0.5*population):
                probability.append(3)
            elif i < int(0.75*population):
                probability.append(2)
            else:
                probability.append(1)
        return probability

    def select(self, inGeneration, probability):
        ## For new selection method
        # selection = inGeneration[:int(len(inGeneration)/4)]
        # selection += random.choicesinGeneration[int(len(inGeneration)/4):], weights = probability, k = len(inGeneration[int(len(inGeneration)/4):]))
        selection = random.choices(inGeneration, weights = probability, k = len(inGeneration))
        return selection

class Test():
    rows = []
    def __init__(self):
        self.test = []

    def getRows(self, fileName):
        file = open(fileName, 'r')
        entire = csv.reader(file)
        for row in entire:
            Test.rows.append(row)

    def getInputs(self, row):
        result = []
        for str in row[:-1]:
            result.append(str == "TRUE")
        return result

    def getWantedOutput(self, row):
        return (row[-1] == "TRUE")

    def getEachResult(self,input1, input2, operator):
        operation = {
            "AND": (input1 and input2),
            "NAND": (not(input1 and input2)),
            "OR": (input1 or input2),
            "NOR": (not(input1 or input2)),
            "XOR": (input1 ^ input2),
            "XNOR": (not(input1 ^ input2))
        }
        return operation.get(operator)

    def getResultedOutput(self, row, operators):
        inputs = self.getInputs(row)
        outputs = []
        for i in range(len(operators)):
            if i == 0:
                outputs.append(self.getEachResult(inputs[i], inputs[i+1], operators[i]))
            else:
                outputs.append(self.getEachResult(outputs[i-1], inputs[i+1], operators[i]))
        return outputs[-1]

    def isSame(self, resultedOutput):
        return resultedOutput == self.getWantedOutput()

def main():
    startTime = time.time()
    inData = Test()
    inData.getRows("truth_table.csv")
    testSize = len(Test.rows) - 1
    targetLen = len(inData.getInputs(Test.rows[0])) - 1 
    solved = False
    generation = Generation()
    for i in range (population):
        rand = Chromosome.createRandom(targetLen)
        ch = Chromosome(rand)
        generation.add(ch)
        generation.sort()
    generationNum = 1
    stuck = False
    stuckCount = 0
    while not solved:
        #Check For Answer
        if (generation.sortedGeneration[0].fitness == testSize):
            solved = True
            print("Proper Chromosome:: ", generation.sortedGeneration[0].chromosome)
            print("Generation count:: ", generationNum)
            endTime = time.time()
            totalTime = endTime - startTime
            print("Time : %f s" % (totalTime))
            break
        generationNum += 1
        # print("A::", generation.sortedGeneration[0].fitness)

        #New Generation
        newGeneration = Generation()

        #Selection
        probabilityList = generation.calculateProbability()
        selectedChromosomes = generation.select(generation.sortedGeneration, probabilityList)

        #Crossover
        crossoverCh = []
        for i in range (0,len(selectedChromosomes),2):
            (newCh1, newCh2) = selectedChromosomes[i].crossover(selectedChromosomes[i+1], targetLen)
            crossoverCh.append(newCh1)
            crossoverCh.append(newCh2)
        
        #Mutation
        mutationCh = []
        for ch in (crossoverCh):
            newCh = ch.mutation(testSize, stuck)
            mutationCh.append(newCh)

        stuck = False
        #Make ne generation
        for i in range(population):
            newGeneration.add(mutationCh[i])

        #Check for stucking
        lastBest = generation.sortedGeneration[0].fitness
        generation.sortedGeneration = []
        while(not(newGeneration.empty())):
            generation.sortedGeneration.append(newGeneration.remove())
        newBest = generation.sortedGeneration[0].fitness
        if newBest == lastBest:
            stuckCount += 1
            if stuckCount > 2:
                # print("STUCK!!")
                stuck = True
                stuckCount = 0
    
main()
