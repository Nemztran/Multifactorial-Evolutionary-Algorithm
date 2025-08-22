# MFEA + GLS + FLS Hybrid Algorithm

## Overview

This directory contains an **innovative hybrid algorithm** that combines:

1. **Multi-Factorial Evolutionary Algorithm (MFEA)** - Multi-task optimization framework
2. **Guided Local Search (GLS)** - Penalty-based escape from local optima  
3. **Fast Local Search (FLS)** - Efficient 2-opt improvement with activation mechanisms

## Algorithm Innovation

### **Key Improvements over Standard MFEA:**

#### 1. **Local Search Integration**
- Apply GLS+FLS to the **best individuals** of each task during evolution
- Maintains population diversity while intensifying search on promising solutions
- **Selective enhancement**: Only improve elite solutions to avoid computational overhead

#### 2. **Guided Local Search (GLS)**
- **Penalty mechanism**: Add penalties to overused edges to escape local optima
- **Utility-based selection**: Penalize edges with highest `cost/(1+penalty)` ratio
- **Dynamic lambda**: Adapt penalty weight based on average edge cost

#### 3. **Fast Local Search (FLS)**  
- **Activation bits**: Track which cities need local improvement
- **Efficient 2-opt**: Only explore moves involving activated cities
- **Incremental improvement**: Continue until no activated cities remain

## Technical Details

### **Hybrid Architecture**
```cpp
for each generation:
    // Standard MFEA operations
    offspring = generateOffspring(current_population)
    
    // GLS+FLS Enhancement (Innovation!)
    best_TSP = findBestIndividual(TSP_task)
    best_TRP = findBestIndividual(TRP_task)
    
    improved_TSP = runGLS_FLS(best_TSP.tour, TSP_matrix, GLS_iterations, alpha)
    improved_TRP = runGLS_FLS(best_TRP.tour, TRP_matrix, GLS_iterations, alpha)
    
    // Update with improved solutions
    current_population = updatePopulation(current_pop, offspring)
```

### **GLS Enhancement Algorithm**
```cpp
1. Initialize penalties = 0 for all edges
2. For each GLS iteration:
   a. Apply FLS with augmented cost: cost + lambda * penalty
   b. Calculate utility for each edge: cost/(1 + penalty)  
   c. Increase penalty for edges with maximum utility
   d. Activate neighborhoods around penalized edges
3. Return best tour found
```

### **FLS 2-opt with Activation**
```cpp
1. Initialize activation bits for all cities
2. While any city is activated:
   a. For each activated city:
      - Try 2-opt moves involving this city
      - If improvement found: activate affected cities
      - Otherwise: deactivate this city
   b. Continue until no improvements possible
```

## Implementation Files

- `TSP_TRPGLS.cpp` - Main hybrid MFEA+GLS+FLS algorithm
- `fls_gls_2opt.h` - GLS and FLS implementation  
- `gls_2opt.h` - Alternative GLS implementation

## Key Parameters

```cpp
const int POP_SIZE = 500;           // Population size
const int MAX_GENERATIONS = 500;    // Evolution generations  
const int GLS_ITER = 100;           // GLS iterations per enhancement
const double alpha = 0.3;           // Penalty weight factor
const double rmp = 0.4;             // Random mating probability
```

## Algorithm Advantages

### **1. Exploitation vs Exploration Balance**
- **MFEA**: Provides exploration through multi-task evolution
- **GLS**: Guides exploitation away from local optima
- **FLS**: Intensive local exploitation with efficient search

### **2. Computational Efficiency**
- Only enhance **elite individuals** (not entire population)
- **Activation mechanism** reduces FLS search space
- **Incremental improvement** stops when no progress possible

### **3. Multi-task Synergy**
- **Knowledge transfer** between TSP and TRP tasks
- **Shared representation** benefits both problems
- **Independent enhancement** preserves task-specific improvements

## Experimental Results

The hybrid algorithm shows significant improvements over standard MFEA:

- **Better convergence** due to local search intensification
- **Escape capability** from local optima via GLS penalties  
- **Faster improvement** on elite solutions
- **Maintained diversity** through selective enhancement

## Usage

```bash
# Compile
make gls

# Run with input
./gls < input.txt

# Input format:
# <dimension>
# <TSP_distance_matrix>
# <TRP_time_matrix>
```

## Research Contribution

This hybrid represents a novel approach to **multi-factorial optimization** by:

1. **Selective local search**: Enhance only elite individuals
2. **Penalty-guided exploration**: Use GLS to escape local optima
3. **Efficient neighborhood search**: FLS with activation mechanisms
4. **Multi-task integration**: Preserve MFEA's knowledge transfer benefits

The combination creates a powerful optimization framework suitable for complex multi-objective combinatorial problems.

## References

1. Gupta, A., et al. (2016). Multifactorial evolution: toward evolutionary multitasking. IEEE TEI.
2. Voudouris, C., & Tsang, E. (1999). Guided local search and its application to the traveling salesman problem. European Journal of Operational Research.
3. Bentley, J. J. (1992). Fast algorithms for geometric traveling salesman problems. ORSA Journal on Computing.
