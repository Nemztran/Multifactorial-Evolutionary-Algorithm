# Multi-Factorial Evolutionary Algorithm (MFEA) for TSP and TRP

## Overview

This repository implements the **Multi-Factorial Evolutionary Algorithm (MFEA)** to solve two optimization problems simultaneously:

1. **TSP (Traveling Salesman Problem)**: Find the shortest tour that visits all cities exactly once
2. **TRP (Time Routing Problem)**: Optimize cumulative travel time routing

## Algorithm Features

- **Multi-task optimization**: Single population solves both problems concurrently
- **Knowledge transfer**: Solutions share information across tasks to improve optimization efficiency
- **Adaptive skill factor**: Each individual specializes in the task where it performs best
- **Simulated Binary Crossover (SBX)**: Robust crossover operator for real-valued genes
- **Rank-based selection**: Maintains diversity while promoting good solutions

## Technical Details

### Algorithm Parameters
- Population size: 200
- Maximum generations: 300
- Random mating probability (RMP): 0.4
- Mutation rate: 0.05
- Distribution index for crossover: 1.0

### Key Components

#### Individual Representation
```cpp
struct Individual {
    vector<double> genes;           // Real-valued chromosome [0,1]
    int skillFactor;               // Best task (0=TSP, 1=TRP)
    vector<double> factorialCost;   // Fitness for each task
    vector<int> factorialRank;      // Rank in population for each task
    double scalarFitness;          // Overall fitness for selection
};
```

#### Fitness Functions
- **TSP**: Minimize total tour distance
- **TRP**: Minimize cumulative travel time with prefix accumulation

#### Genetic Operators
- **Encoding**: Real-valued genes decoded to permutations via rank-sorting
- **Crossover**: Simulated Binary Crossover (SBX) when parents have same skill or random mating occurs
- **Mutation**: Gaussian perturbation with bounds checking
- **Selection**: Elitist selection based on scalar fitness

## Input Format

```
<TSP_dimension> <TRP_dimension>
<TSP_distance_matrix>
<TRP_time_matrix>
```

### Example Input
```
4 3
0 10 15 20
10 0 35 25
15 35 0 30
20 25 30 0
0 5 8
5 0 12
8 12 0
```

## Output Format

```
<best_TSP_cost> <best_TRP_cost>
```

## Compilation and Usage

### Prerequisites
- C++11 or later
- Standard library support for `<random>`, `<algorithm>`, etc.

### Compilation
```bash
g++ -std=c++11 -O2 TRP__TSP.cpp -o mfea
```

### Execution
```bash
./mfea < input.txt
```

## Algorithm Workflow

1. **Initialization**: Generate random population, evaluate on both tasks
2. **Evolution Loop**:
   - **Parent Selection**: Random selection from population
   - **Reproduction**: 
     - If same skill or random < RMP: Apply crossover
     - Otherwise: Apply mutation only
   - **Evaluation**: Evaluate offspring on inherited skill task
   - **Environmental Selection**: Combine populations, select best individuals
3. **Termination**: Output best solutions found for each task

## Key Innovation: Knowledge Transfer

The MFEA enables knowledge transfer between tasks through:
- **Shared representation**: Single chromosome encodes solutions for both problems
- **Cross-task mating**: Individuals from different tasks can reproduce
- **Implicit transfer**: Good building blocks discovered for one task may benefit the other

## Performance Characteristics

- **Convergence**: Typically converges within 200-300 generations
- **Scalability**: Handles problems with different dimensions efficiently
- **Robustness**: Maintains solution quality across multiple runs

## File Structure

```
MFEA/
├── TRP__TSP.cpp          # Main implementation
├── README.md             # This documentation
├── Makefile              # Build configuration
├── input_example.txt     # Sample input
└── results/              # Output directory
```

## References

1. Gupta, A., Ong, Y. S., & Feng, L. (2016). Multifactorial evolution: toward evolutionary multitasking. IEEE transactions on evolutionary computation, 20(3), 343-357.

2. Bali, K. K., Ong, Y. S., Gupta, A., & Tan, P. S. (2017). Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II. IEEE Transactions on Evolutionary Computation, 24(1), 69-83.

## License

This project is developed for educational purposes as part of IT3020 - Discrete Optimization course.

## Author

[Your Name]  
[Your University]  
[Date: August 2025]
