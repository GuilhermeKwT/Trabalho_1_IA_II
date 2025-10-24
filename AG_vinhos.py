import random

from matplotlib import pyplot as plt

class vinho :
    def __init__(self, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        
def fitness(x):
    return 55.16+0.0677*(x.fixed_acidity)-1.3279*(x.volatile_acidity)-0.1097*(x.citric_acid)+0.0436*(x.residual_sugar)-0.4837*(x.chlorides)+0.0060*(x.free_sulfur_dioxide)-0.0025*(x.total_sulfur_dioxide)-54.97*(x.density)+0.4393*(x.pH)+0.7683*(x.sulphates)+0.2670*(x.alcohol)

def init_population(pop_size, gene_bounds):
    population = []
    for _ in range(pop_size):
        individual = vinho(
            fixed_acidity = random.uniform(gene_bounds['fixed_acidity'][0], gene_bounds['fixed_acidity'][1]),
            volatile_acidity = random.uniform(gene_bounds['volatile_acidity'][0], gene_bounds['volatile_acidity'][1]),
            citric_acid = random.uniform(gene_bounds['citric_acid'][0], gene_bounds['citric_acid'][1]),
            residual_sugar = random.uniform(gene_bounds['residual_sugar'][0], gene_bounds['residual_sugar'][1]),
            chlorides = random.uniform(gene_bounds['chlorides'][0], gene_bounds['chlorides'][1]),
            free_sulfur_dioxide = random.uniform(gene_bounds['free_sulfur_dioxide'][0], gene_bounds['free_sulfur_dioxide'][1]),
            total_sulfur_dioxide = random.uniform(gene_bounds['total_sulfur_dioxide'][0], gene_bounds['total_sulfur_dioxide'][1]),
            density = random.uniform(gene_bounds['density'][0], gene_bounds['density'][1]),
            pH = random.uniform(gene_bounds['pH'][0], gene_bounds['pH'][1]),
            sulphates = random.uniform(gene_bounds['sulphates'][0], gene_bounds['sulphates'][1]),
            alcohol = random.uniform(gene_bounds['alcohol'][0], gene_bounds['alcohol'][1])
        )
        population.append(individual)
    return population

def select_parents(population, fitnesses, num_parents):
    selected_parents = random.choices(
        population,
        weights=fitnesses,
        k=num_parents
    )
    return selected_parents

def crossover(parent1, parent2):
    child1 = vinho(
        fixed_acidity=(random.choice([parent1.fixed_acidity, parent2.fixed_acidity])),
        volatile_acidity=(random.choice([parent1.volatile_acidity, parent2.volatile_acidity])),
        citric_acid=(random.choice([parent1.citric_acid, parent2.citric_acid])),
        residual_sugar=(random.choice([parent1.residual_sugar, parent2.residual_sugar])),
        chlorides=(random.choice([parent1.chlorides, parent2.chlorides])),
        free_sulfur_dioxide=(random.choice([parent1.free_sulfur_dioxide, parent2.free_sulfur_dioxide])),
        total_sulfur_dioxide=(random.choice([parent1.total_sulfur_dioxide, parent2.total_sulfur_dioxide])),
        density=(random.choice([parent1.density, parent2.density])),
        pH=(random.choice([parent1.pH, parent2.pH])),
        sulphates=(random.choice([parent1.sulphates, parent2.sulphates])),
        alcohol=(random.choice([parent1.alcohol, parent2.alcohol]))
    )
    child2 = vinho(
        fixed_acidity=(random.choice([parent1.fixed_acidity, parent2.fixed_acidity])),
        volatile_acidity=(random.choice([parent1.volatile_acidity, parent2.volatile_acidity])),
        citric_acid=(random.choice([parent1.citric_acid, parent2.citric_acid])),
        residual_sugar=(random.choice([parent1.residual_sugar, parent2.residual_sugar])),
        chlorides=(random.choice([parent1.chlorides, parent2.chlorides])),
        free_sulfur_dioxide=(random.choice([parent1.free_sulfur_dioxide, parent2.free_sulfur_dioxide])),
        total_sulfur_dioxide=(random.choice([parent1.total_sulfur_dioxide, parent2.total_sulfur_dioxide])),
        density=(random.choice([parent1.density, parent2.density])),
        pH=(random.choice([parent1.pH, parent2.pH])),
        sulphates=(random.choice([parent1.sulphates, parent2.sulphates])),
        alcohol=(random.choice([parent1.alcohol, parent2.alcohol]))
    )

    if (fitness(child1) < fitness(parent1)) and (fitness(child1) < fitness(parent2)):
        child1 = parent1 if fitness(parent1) > fitness(parent2) else parent2
    
    if (fitness(child2) < fitness(parent1)) and (fitness(child2) < fitness(parent2)):
        child2 = parent1 if fitness(parent1) > fitness(parent2) else parent2
    
    return child1, child2

def mutate(individual, gene_bounds):
    gene = random.randint(0, 10)
    if gene == 0:
        individual.fixed_acidity = random.uniform(gene_bounds['fixed_acidity'][0], gene_bounds['fixed_acidity'][1])
    elif gene == 1:
        individual.volatile_acidity = random.uniform(gene_bounds['volatile_acidity'][0], gene_bounds['volatile_acidity'][1])
    elif gene == 2:
        individual.citric_acid = random.uniform(gene_bounds['citric_acid'][0], gene_bounds['citric_acid'][1])
    elif gene == 3:
        individual.residual_sugar = random.uniform(gene_bounds['residual_sugar'][0], gene_bounds['residual_sugar'][1])
    elif gene == 4:
        individual.chlorides = random.uniform(gene_bounds['chlorides'][0], gene_bounds['chlorides'][1])
    elif gene == 5:
        individual.free_sulfur_dioxide = random.uniform(gene_bounds['free_sulfur_dioxide'][0], gene_bounds['free_sulfur_dioxide'][1])
    elif gene == 6:
        individual.total_sulfur_dioxide = random.uniform(gene_bounds['total_sulfur_dioxide'][0], gene_bounds['total_sulfur_dioxide'][1])
    elif gene == 7:
        individual.density = random.uniform(gene_bounds['density'][0], gene_bounds['density'][1])
    elif gene == 8:
        individual.pH = random.uniform(gene_bounds['pH'][0], gene_bounds['pH'][1])
    elif gene == 9:
        individual.sulphates = random.uniform(gene_bounds['sulphates'][0], gene_bounds['sulphates'][1])
    elif gene == 10:
        individual.alcohol = random.uniform(gene_bounds['alcohol'][0], gene_bounds['alcohol'][1])
        
    return individual

def genetic_algorithm(pop_size, gene_bounds, num_generations, mutation_rate, crossover_rate):
    population = init_population(pop_size, gene_bounds)
    best_fitness_history = []
    avg_fitness_history = []
    
    for generation in range(num_generations):
        fitnesses = [fitness(ind) for ind in population]
        
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(best_fitness)    # Adicionar à lista
        avg_fitness_history.append(avg_fitness) 
        
        new_population = []
        while len(new_population) < pop_size:
            parents = select_parents(population, fitnesses, 2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parents[0], parents[1])
            else:
                child1, child2 = parents[0], parents[1]
            if random.random() < mutation_rate:
                child1 = mutate(child1, gene_bounds)
            if random.random() < mutation_rate:
                child2 = mutate(child2, gene_bounds)
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
        
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {generation+1}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")
    
    best_individual = max(population, key=fitness)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_generations + 1), best_fitness_history, 'b-', label='Best Fitness')
    plt.plot(range(1, num_generations + 1), avg_fitness_history, 'r-', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_individual

gene_bounds = {
    'fixed_acidity': (4.6, 15.9),
    'volatile_acidity': (0.12, 1.58),
    'citric_acid': (0.0, 1.0),
    'residual_sugar': (0.9, 15.5),
    'chlorides': (0.012, 0.611),
    'free_sulfur_dioxide': (1.0, 72.0),
    'total_sulfur_dioxide': (6.0, 289.0),
    'density': (0.99007, 1.00369),
    'pH': (2.74, 4.01),
    'sulphates': (0.33, 2.0),
    'alcohol': (8.0, 14.9)
}

best_wine = genetic_algorithm(pop_size=150, gene_bounds=gene_bounds, num_generations=100, mutation_rate=0.1, crossover_rate=0.8)
print("Best Wine Parameters:")
print(f"Fixed Acidity: {best_wine.fixed_acidity}")
print(f"Volatile Acidity: {best_wine.volatile_acidity}")
print(f"Citric Acid: {best_wine.citric_acid}")
print(f"Residual Sugar: {best_wine.residual_sugar}")
print(f"Chlorides: {best_wine.chlorides}")
print(f"Free Sulfur Dioxide: {best_wine.free_sulfur_dioxide}")
print(f"Total Sulfur Dioxide: {best_wine.total_sulfur_dioxide}")
print(f"Density: {best_wine.density}")
print(f"pH: {best_wine.pH}")
print(f"Sulphates: {best_wine.sulphates}")
print(f"Alcohol: {best_wine.alcohol}")