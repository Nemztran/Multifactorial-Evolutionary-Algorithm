/**
 * Multi-Factorial Evolutionary Algorithm (MFEA) for TSP and TRP
 * 
 * This implementation solves two optimization problems simultaneously:
 * 1. TSP (Traveling Salesman Problem): Find shortest tour visiting all cities
 * 2. TRP (Time Routing Problem): Optimize cumulative travel time
 * 
 * The algorithm uses knowledge transfer between tasks to improve optimization efficiency.
 * 
 * Author: [Your Name]
 * Date: August 2025
 * Course: IT3020 - Discrete Optimization
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iomanip>

using namespace std;

// Algorithm parameters
const int POP_SIZE = 200;           // Population size
const int MAX_GENERATIONS = 300;    // Maximum number of generations
const double RMP = 0.4;             // Random mating probability
const double MUTATION_RATE = 0.05;  // Mutation rate
const double ETA_C = 1.0;           // Distribution index for crossover

// Problem dimensions
int problemDim_TSP, problemDim_TRP;
int problemDim;
/**
 * TSP Problem Structure
 * Contains distance matrix for Traveling Salesman Problem
 */
struct TSP {
    vector<vector<double>> distance;
};

/**
 * TRP Problem Structure  
 * Contains time matrix for Time Routing Problem
 */
struct TRP {
    vector<vector<double>> time;
};

/**
 * Combined Problem Structure
 * Holds both TSP and TRP problem instances
 */
struct Problem {
    TSP tsp;
    TRP trp;
};

Problem problems;
/**
 * Initialize problem instances by reading input data
 * Reads distance matrix for TSP and time matrix for TRP
 */
void initializeProblems() {
    // Initialize TSP distance matrix
    problems.tsp.distance.resize(problemDim_TSP, vector<double>(problemDim_TSP));
    
    // Initialize TRP time matrix
    problems.trp.time.resize(problemDim_TRP, vector<double>(problemDim_TRP));
    
    // Read TSP distance matrix
    cout << "Reading TSP distance matrix (" << problemDim_TSP << "x" << problemDim_TSP << ")..." << endl;
    for (int i = 0; i < problemDim_TSP; i++) {
        for (int j = 0; j < problemDim_TSP; j++) {
            cin >> problems.tsp.distance[i][j];
        }
    }
    
    // Read TRP time matrix
    cout << "Reading TRP time matrix (" << problemDim_TRP << "x" << problemDim_TRP << ")..." << endl;
    for (int i = 0; i < problemDim_TRP; i++) {
        for (int j = 0; j < problemDim_TRP; j++) {
            cin >> problems.trp.time[i][j];
        }
    }
}
/**
 * Generate random genes for initialization
 * @return Vector of random doubles in [0,1]
 */
vector<double> generateRandomGenes() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    vector<double> genes(problemDim);
    for (int i = 0; i < problemDim; i++) {
        genes[i] = dis(gen);
    }
    return genes;
}
/**
 * Decode genes into permutation using rank-based mapping
 * @param problemType Either "TSP" or "TRP"
 * @param genes Vector of real-valued genes
 * @return Permutation vector for the specified problem
 */
vector<int> decodeGenes(const string& problemType, vector<double> genes) {
    int dimension = (problemType == "TRP") ? problemDim_TRP : problemDim_TSP;
    genes.resize(dimension);
    
    vector<double> sortedGenes = genes;
    sort(sortedGenes.begin(), sortedGenes.end());
    
    vector<int> permutation(dimension);
    for (int i = 0; i < dimension; i++) {
        auto it = find(sortedGenes.begin(), sortedGenes.end(), genes[i]);
        permutation[i] = distance(sortedGenes.begin(), it);
    }
    
    return permutation;
}
/**
 * Individual structure for MFEA
 * Represents a solution that can be evaluated on multiple tasks
 */
struct Individual {
    vector<double> genes;              // Real-valued chromosome representation
    int skillFactor;                   // Task this individual performs best on (0=TSP, 1=TRP)
    vector<double> factorialCost;      // Fitness values for each task [TSP_cost, TRP_cost]
    vector<int> factorialRank;         // Rank in population for each task
    double scalarFitness;              // Scalar fitness for selection
    
    // Constructor
    Individual() {
        genes.resize(problemDim);
        factorialCost.resize(2);
        factorialRank.resize(2);
        skillFactor = 0;
        scalarFitness = 0.0;
    }
};
/**
 * Calculate factorial cost (fitness) for a given problem
 * @param genes Individual's genes
 * @param problemType Either "TSP" or "TRP" 
 * @param problems Problem instances
 * @return Fitness value for the specified problem
 */
double calculateFactorialCost(const vector<double>& genes, const string& problemType, const Problem& problems) {
    vector<int> permutation = decodeGenes(problemType, genes);
    
    if (problemType == "TRP") {
        // TRP: Calculate cumulative travel time
        double totalTime = 0.0;
        double cumulativeTime = 0.0;
        
        for (int i = 0; i < problemDim_TRP - 1; i++) {
            double segmentTime = problems.trp.time[permutation[i]][permutation[i + 1]];
            totalTime += cumulativeTime + segmentTime;
            cumulativeTime += segmentTime;
        }
        return totalTime;
    } else {
        // TSP: Calculate total tour distance
        double totalDistance = 0.0;
        for (int i = 0; i < problemDim_TSP; i++) {
            int current = permutation[i];
            int next = permutation[(i + 1) % problemDim_TSP];
            totalDistance += problems.tsp.distance[current][next];
        }
        return totalDistance;
    }
}
/**
 * Calculate factorial rank of an individual for a specific task
 * @param population Current population
 * @param individual Individual to rank
 * @param problemType Task to evaluate ("TSP" or "TRP")
 * @return Rank (1-based) in the population for the specified task
 */
int calculateFactorialRank(const vector<Individual>& population, const Individual& individual, const string& problemType) {
    vector<double> fitnessValues(population.size());
    int taskIndex = (problemType == "TSP") ? 0 : 1;
    
    for (size_t i = 0; i < population.size(); i++) {
        fitnessValues[i] = population[i].factorialCost[taskIndex];
    }
    
    sort(fitnessValues.begin(), fitnessValues.end());
    auto it = find(fitnessValues.begin(), fitnessValues.end(), individual.factorialCost[taskIndex]);
    return distance(fitnessValues.begin(), it) + 1;
}
/**
 * Initialize population for MFEA
 * Creates random individuals and evaluates them on both tasks
 * @return Initial population
 */
vector<Individual> initializePopulation() {
    vector<Individual> population(POP_SIZE);
    
    cout << "Initializing population of size " << POP_SIZE << "..." << endl;
    
    // Generate random individuals and evaluate them
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].genes = generateRandomGenes();
        population[i].factorialCost[0] = calculateFactorialCost(population[i].genes, "TSP", problems);
        population[i].factorialCost[1] = calculateFactorialCost(population[i].genes, "TRP", problems);
    }
    
    // Calculate skill factors and ranks
    for (int i = 0; i < POP_SIZE; i++) {
        // Skill factor: task with better relative performance
        population[i].skillFactor = (population[i].factorialCost[0] > abs(population[i].factorialCost[1])) ? 1 : 0;
        
        population[i].factorialRank[0] = calculateFactorialRank(population, population[i], "TSP");
        population[i].factorialRank[1] = calculateFactorialRank(population, population[i], "TRP");
        
        // Scalar fitness: inverse of best rank
        population[i].scalarFitness = 1.0 / min(population[i].factorialRank[0], population[i].factorialRank[1]);
    }
    
    cout << "Population initialized successfully!" << endl;
    return population;
}
/**
 * Simulated Binary Crossover (SBX)
 * @param parent1 First parent
 * @param parent2 Second parent
 * @param child1 First offspring (output)
 * @param child2 Second offspring (output)
 */
void simulatedBinaryCrossover(const Individual& parent1, const Individual& parent2, 
                             Individual& child1, Individual& child2) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    double u = dis(gen);
    double beta;
    
    if (u <= 0.5) {
        beta = pow(2.0 * u, 1.0 / (ETA_C + 1.0));
    } else {
        beta = pow(2.0 * (1.0 - u), -1.0 / (ETA_C + 1.0));
    }
    
    for (int i = 0; i < problemDim; i++) {
        child1.genes[i] = 0.5 * ((1.0 + beta) * parent1.genes[i] + (1.0 - beta) * parent2.genes[i]);
        child2.genes[i] = 0.5 * ((1.0 - beta) * parent1.genes[i] + (1.0 + beta) * parent2.genes[i]);
        
        // Ensure genes stay in [0,1] bounds
        child1.genes[i] = max(0.0, min(1.0, child1.genes[i]));
        child2.genes[i] = max(0.0, min(1.0, child2.genes[i]));
    }
}
/**
 * Apply slight mutation to an individual
 * @param parent Parent individual
 * @param child Child individual (output)
 */
void applyMutation(const Individual& parent, Individual& child) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dis(0.0, MUTATION_RATE);
    
    for (int i = 0; i < problemDim; i++) {
        child.genes[i] = parent.genes[i] + dis(gen);
        // Ensure genes stay in [0,1] bounds
        child.genes[i] = max(0.0, min(1.0, child.genes[i]));
    }
}
/**
 * Evaluate a child individual on a single task
 * @param child Child to evaluate
 * @param parent Parent (for skill factor inheritance)
 */
void evaluateOnSingleTask(Individual& child, const Individual& parent) {
    int taskIndex = parent.skillFactor;
    string problemType = (taskIndex == 0) ? "TSP" : "TRP";
    
    child.factorialCost[taskIndex] = calculateFactorialCost(child.genes, problemType, problems);
    
    // Set infinite cost for the other task (not evaluated)
    int otherTask = 1 - taskIndex;
    child.factorialCost[otherTask] = numeric_limits<double>::infinity();
    
    child.skillFactor = taskIndex;
}
/**
 * Generate offspring population
 * @param currentPopulation Current parent population
 * @return Offspring population
 */
vector<Individual> generateOffspring(vector<Individual>& currentPopulation) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> realDist(0.0, 1.0);
    uniform_int_distribution<> intDist(0, POP_SIZE - 1);
    
    vector<Individual> offspring(POP_SIZE);
    
    for (int i = 0; i < POP_SIZE / 2; i++) {
        // Select two parents randomly
        int indexA, indexB;
        do {
            indexA = intDist(gen);
            indexB = intDist(gen);
        } while (indexA == indexB);
        
        Individual parentA = currentPopulation[indexA];
        Individual parentB = currentPopulation[indexB];
        
        double randomValue = realDist(gen);
        
        // Check if crossover should occur
        if (parentA.skillFactor == parentB.skillFactor || randomValue < RMP) {
            // Perform crossover
            simulatedBinaryCrossover(parentA, parentB, offspring[2 * i], offspring[2 * i + 1]);
            
            // Randomly assign skill factors
            double skill1 = realDist(gen);
            double skill2 = realDist(gen);
            
            if (skill1 < 0.5) {
                evaluateOnSingleTask(offspring[2 * i], parentA);
            } else {
                evaluateOnSingleTask(offspring[2 * i], parentB);
            }
            
            if (skill2 < 0.5) {
                evaluateOnSingleTask(offspring[2 * i + 1], parentA);
            } else {
                evaluateOnSingleTask(offspring[2 * i + 1], parentB);
            }
        } else {
            // Perform mutation only
            applyMutation(parentA, offspring[2 * i]);
            evaluateOnSingleTask(offspring[2 * i], parentA);
            
            applyMutation(parentB, offspring[2 * i + 1]);
            evaluateOnSingleTask(offspring[2 * i + 1], parentB);
        }
    }
    
    return offspring;
}
/**
 * Comparator for sorting individuals by scalar fitness (descending)
 */
bool compareScalarFitness(const Individual& a, const Individual& b) {
    return a.scalarFitness > b.scalarFitness;
}

/**
 * Update population by combining current and offspring populations
 * @param currentPopulation Current population
 * @param offspringPopulation Offspring population
 * @return Updated population (best POP_SIZE individuals)
 */
vector<Individual> updatePopulation(const vector<Individual>& currentPopulation, 
                                   const vector<Individual>& offspringPopulation) {
    // Combine current and offspring populations
    vector<Individual> combinedPopulation = currentPopulation;
    combinedPopulation.insert(combinedPopulation.end(), 
                             offspringPopulation.begin(), offspringPopulation.end());
    
    // Recalculate skill factors, ranks, and scalar fitness
    for (size_t i = 0; i < combinedPopulation.size(); i++) {
        combinedPopulation[i].skillFactor = 
            (combinedPopulation[i].factorialCost[0] > abs(combinedPopulation[i].factorialCost[1])) ? 1 : 0;
        
        combinedPopulation[i].factorialRank[0] = calculateFactorialRank(combinedPopulation, combinedPopulation[i], "TSP");
        combinedPopulation[i].factorialRank[1] = calculateFactorialRank(combinedPopulation, combinedPopulation[i], "TRP");
        
        combinedPopulation[i].scalarFitness = 1.0 / min(combinedPopulation[i].factorialRank[0], 
                                                        combinedPopulation[i].factorialRank[1]);
    }
    
    // Sort by scalar fitness and select best POP_SIZE individuals
    sort(combinedPopulation.begin(), combinedPopulation.end(), compareScalarFitness);
    
    return vector<Individual>(combinedPopulation.begin(), combinedPopulation.begin() + POP_SIZE);
}


/**
 * Main MFEA algorithm
 */
int main() {
    cout << "=== Multi-Factorial Evolutionary Algorithm (MFEA) ===" << endl;
    cout << "Solving TSP and TRP simultaneously" << endl;
    cout << "======================================================" << endl;
    
    // Read problem dimensions
    cout << "Enter TSP dimension and TRP dimension: ";
    cin >> problemDim_TSP >> problemDim_TRP;
    problemDim = max(problemDim_TSP, problemDim_TRP);
    
    cout << "TSP dimension: " << problemDim_TSP << endl;
    cout << "TRP dimension: " << problemDim_TRP << endl;
    cout << "Gene dimension: " << problemDim << endl;
    
    // Initialize problems
    initializeProblems();
    
    // Initialize population
    vector<Individual> currentPopulation = initializePopulation();
    
    cout << "\n=== Evolution Process ===" << endl;
    cout << "Population size: " << POP_SIZE << endl;
    cout << "Max generations: " << MAX_GENERATIONS << endl;
    cout << "Random mating probability: " << RMP << endl;
    
    // Evolution loop
    for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
        // Generate offspring
        vector<Individual> offspringPopulation = generateOffspring(currentPopulation);
        
        // Update population
        currentPopulation = updatePopulation(currentPopulation, offspringPopulation);
        
        // Progress report
        if (generation % 50 == 0) {
            cout << "Generation " << generation << " completed" << endl;
        }
    }
    
    cout << "\n=== Results ===" << endl;
    
    // Find best solutions
    double bestTSP = numeric_limits<double>::infinity();
    double bestTRP = numeric_limits<double>::infinity();
    vector<int> bestTSPTour, bestTRPRoute;
    
    for (int i = 0; i < POP_SIZE; i++) {
        if (currentPopulation[i].factorialCost[0] < bestTSP) {
            bestTSP = currentPopulation[i].factorialCost[0];
            bestTSPTour = decodeGenes("TSP", currentPopulation[i].genes);
        }
        if (currentPopulation[i].factorialCost[1] < bestTRP) {
            bestTRP = currentPopulation[i].factorialCost[1];
            bestTRPRoute = decodeGenes("TRP", currentPopulation[i].genes);
        }
    }
    
    cout << fixed << setprecision(6);
    cout << "Best TSP cost: " << bestTSP << endl;
    cout << "Best TRP cost: " << bestTRP << endl;
    
    cout << "\nBest TSP tour: ";
    for (int city : bestTSPTour) {
        cout << city << " ";
    }
    cout << endl;
    
    cout << "Best TRP route: ";
    for (int node : bestTRPRoute) {
        cout << node << " ";
    }
    cout << endl;
    
    // Output for automatic testing
    cout << bestTSP << " " << bestTRP << endl;
    
    return 0;
}
