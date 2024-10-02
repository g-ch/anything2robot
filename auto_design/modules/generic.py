import random
import math
class Generic_Algorithm():

    def __init__(self, bounds, int_bounds, genome_length=1000, # 根据范围计算genome_length
                                           generation_num=10,
                                           population_size=100,
                                           mutation_rate=0.01,
                                           crossover_rate=0.3,
                                           mutate_ratio=0.3) -> None:
        self.bounds = bounds
        self.int_bounds = int_bounds
        self.genome_length = genome_length
        self.generation_num = generation_num
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutate_ratio = mutate_ratio

        # Print scale for each continuous variable
        # integer_segment_length_sum = sum([math.ceil(math.log2(len(int_bound))) for int_bound in self.int_bounds])
        # continuous_segment_length = (genome_length - integer_segment_length_sum) // len(bounds)
        # for i in range(len(self.bounds)):
        #     lower, upper = self.bounds[i]
        #     scale = (upper - lower) / (2**continuous_segment_length - 1)
        #     print(f"Variable {i}: {lower} to {upper}, scale: {scale}")
    
    def fitness_function(self, genome) -> float:
        pass

    def generate_individual(self):
        """Generate a random individual."""
        return [random.randint(0, 1) for _ in range(self.genome_length)]

    def generate_population(self):
        """Generate a population of individuals."""
        return [self.generate_individual() for _ in range(self.population_size)]
    
    def encode(self, variables):
        """
        Encode multiple variables with individual bounds and integer variables with specified options into a binary genome.
        """

        genome = []
        index = 0

        num_vars = len(self.bounds)
        integer_segment_length_sum = sum([math.ceil(math.log2(len(int_bound))) if len(int_bound) > 1 else 1
                                           for int_bound in self.int_bounds])
        continuous_segment_length = (self.genome_length - integer_segment_length_sum) // num_vars

        # Encode continuous variables
        for i in range(num_vars):
            lower, upper = self.bounds[i]
            scale = (upper - lower) / (2**continuous_segment_length - 1)
            int_val = int((variables[i] - lower) / scale)
            binary_val = bin(int_val)[2:].zfill(continuous_segment_length)
            genome.extend([int(bit) for bit in binary_val])

        # Encode integer variables
        for int_bound in self.int_bounds:
            int_segment_length = math.ceil(math.log2(len(int_bound)))
            if int_segment_length == 0:
                int_segment_length = 1
            int_val = int_bound.index(int(variables[num_vars]))
            binary_val = bin(int_val)[2:].zfill(int_segment_length)
            genome.extend([int(bit) for bit in binary_val])

        return genome

    def decode(self, genome):
        """
        Decode a binary genome into multiple variables with individual bounds and integer variables with specified options.
        """
        # Calculate segment length for continuous variables
        num_vars = len(self.bounds)
        integer_segment_length_sum = sum([math.ceil(math.log2(len(int_bound))) if len(int_bound) > 1 else 1
                                           for int_bound in self.int_bounds])
        continuous_segment_length = (len(genome) - integer_segment_length_sum) // num_vars
        
        variables = []
        index = 0

        # Decode continuous variables
        for i in range(num_vars):
            segment = genome[index:index + continuous_segment_length]
            int_val = int("".join(map(str, segment)), 2)
            lower, upper = self.bounds[i]
            scale = (upper - lower) / (2**continuous_segment_length - 1)
            variable = lower + int_val * scale
            variables.append(variable)
            index += continuous_segment_length

        # Decode integer variables
        for int_bound in self.int_bounds:
            # Calculate segment length for each integer variable
            int_segment_length = math.ceil(math.log2(len(int_bound)))
            if int_segment_length == 0:
                int_segment_length = 1
            segment = genome[index:index + int_segment_length]
            int_val = int("".join(map(str, segment)), 2) % len(int_bound)
            variables.append(int_bound[int_val])
            index += int_segment_length

        return variables

    def tournament_selection(self, population, tournament_size=3):
        """Selects one individual using tournament selection."""
        tournament = random.sample(population, tournament_size)
        fittest_individual = min(tournament, key=lambda genome : self.fitness_function(genome))
        return fittest_individual

    def crossover(self, parent1, parent2):
        """Apply crossover to two parents to create two children."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.genome_length - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1[:], parent2[:]

    def mutate(self, genome):
        """Apply mutation to a genome."""
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                genome[i] = 1 - genome[i]
        return genome

    def run_generic(self, initial_population=None, log_list=[]):
        # Initialize population
        if initial_population is not None:
            population = self.initial_population
        else:
            population = self.generate_population()

        # Store log result
        log_result = []
        best_individuals = []

        # Run GA
        for generation in range(self.generation_num):
            # Record the result in log list
            if generation in log_list:
                log_result.append(min(population, key=lambda genome : self.fitness_function(genome)))
            
            # Selection
            new_population = [self.tournament_selection(population) for _ in range(self.population_size)]

            # Crossover
            population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = new_population[i], new_population[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                population.extend([child1, child2])

            # Mutation
            population = [self.mutate(individual) for individual in population if random.random() < self.mutate_ratio]

            # Evaluate the fitness of the new population
            fittest_individual = min(population, key=lambda genome : self.fitness_function(genome))
            best_individuals.append(fittest_individual)

            # 打印一下variance
            print(f"Generation {generation}, Best Fitness {self.fitness_function(fittest_individual)}")

        # Results
        best_individual = min(best_individuals, key=lambda genome : self.fitness_function(genome))
        print(f"Best Fitness: {self.fitness_function(best_individual)}")
        return best_individual, log_result
    
class Improved_Generic_Algorithm(Generic_Algorithm):

    def __init__(self, bounds, int_bounds, genome_length=1000, # 根据范围计算genome_length
                                           generation_num=10,
                                           population_size=100,
                                           mutation_rate=0.01,
                                           crossover_rate=0.3,
                                           mutate_ratio=0.3) -> None:
        super().__init__(bounds, int_bounds, genome_length, generation_num, population_size, mutation_rate, crossover_rate, mutate_ratio)

    def random_select(self, lower, upper):
        # return random.uniform(lower, upper)
        return random.gauss((lower + upper) / 2, (upper - lower) / 6)
    
    def fitness_function(self, genome) -> float:
        pass

    def generate_individual(self):
        """Generate a random individual."""
        # Generate continuous variables that are uniformly distributed within the bounds
        variables = []
        for lower, upper in self.bounds:
            variable = self.random_select(lower, upper)
            variables.append(variable)

        # Generate integer variables that are randomly selected from the options
        for int_bound in self.int_bounds:
            variable = random.choice(int_bound)
            variables.append(variable)
        return self.encode(variables)

    def encode(self, variables):
        return variables
    
    def decode(self, genome):
        return genome
    
    def crossover(self, parent1, parent2):
        """Apply crossover to two parents to create two children."""
        if random.random() < self.crossover_rate:
            choices = [random.choice([0, 1]) for _ in range(len(parent1))]
            child1 = [parent1[i] if choices[i] == 0 else parent2[i] for i in range(len(parent1))]
            child2 = [parent2[i] if choices[i] == 0 else parent1[i] for i in range(len(parent1))]
            return child1, child2
        return parent1[:], parent2[:]
    
    def mutate(self, genome):
        """Apply mutation to a genome."""
        for i in range(len(self.bounds)):
            if random.random() < self.mutation_rate:
                genome[i] = self.random_select(self.bounds[i][0], self.bounds[i][1])
        for i in range(len(self.int_bounds)):
            if random.random() < self.mutation_rate:
                genome[i + len(self.bounds)] = random.choice(self.int_bounds[i])
        return genome
