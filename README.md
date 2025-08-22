# Multi-Factorial Evolutionary Algorithm (MFEA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-11-blue.svg)](https://en.cppreference.com/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](https://github.com/Nemztran/Multifactorial-Evolutionary-Algorithm)

> **Advanced evolutionary algorithm for simultaneous optimization of TSP and TRP problems**

## 🚀 Algorithm Innovation: Hybrid MFEA+GLS+FLS

### **Enhanced Version Available**
The `/GLS` directory contains an **innovative hybrid algorithm** that combines:
- **MFEA** (Multi-Factorial Evolutionary Algorithm)
- **GLS** (Guided Local Search) 
- **FLS** (Fast Local Search)

This hybrid provides significant improvements over standard MFEA through:
- **Selective local search enhancement** of elite individuals
- **Penalty-guided escape** from local optima  
- **Efficient 2-opt improvement** with activation mechanisms
- **Preserved multi-task knowledge transfer**

See `/GLS/README.md` for detailed documentation of this innovation.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Algorithm Details](#-algorithm-details)
- [Input/Output Format](#-inputoutput-format)
- [Usage Examples](#-usage-examples)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **🎯 Multi-task optimization**: Single population solves both problems concurrently
- **🔄 Knowledge transfer**: Solutions share information across tasks to improve efficiency
- **⚡ Adaptive skill factor**: Each individual specializes in optimal task
- **🧬 Advanced operators**: Simulated Binary Crossover (SBX) for robust evolution
- **📊 Rank-based selection**: Maintains diversity while promoting convergence

## 🚀 Quick Start

### Prerequisites
```bash
# C++11 or later
g++ --version
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Nemztran/Multifactorial-Evolutionary-Algorithm.git
cd Multifactorial-Evolutionary-Algorithm

# Compile
make

# Or manually:
g++ -std=c++11 -O2 TRP__TSP.cpp -o mfea
```

### Run Example
```bash
./mfea < input_example.txt
```

## 🔬 Algorithm Details

### 🎯 Problem Domains

1. **TSP (Traveling Salesman Problem)**: Find the shortest tour visiting all cities exactly once
2. **TRP (Time Routing Problem)**: Optimize cumulative travel time routing

### ⚙️ Algorithm Parameters
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

## 📥 Input/Output Format

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

## 💻 Usage Examples

### Basic Usage
```bash
# Run with provided example
./mfea < input_example.txt

# Expected output format
# 85.5 42.3
```

### Custom Input
```bash
# Create your own input file
echo "3 3
0 10 20
10 0 15
20 15 0
0 5 8
5 0 12
8 12 0" > my_input.txt

./mfea < my_input.txt
```

## 📊 Performance

- **⚡ Convergence**: Typically converges within 200-300 generations
- **📈 Scalability**: Handles problems with different dimensions efficiently  
- **🎯 Robustness**: Maintains solution quality across multiple runs
- **🔄 Knowledge Transfer**: Cross-task learning improves overall performance

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

1. Gupta, A., Ong, Y. S., & Feng, L. (2016). Multifactorial evolution: toward evolutionary multitasking. IEEE transactions on evolutionary computation, 20(3), 343-357.

2. Bali, K. K., Ong, Y. S., Gupta, A., & Tan, P. S. (2017). Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II. IEEE Transactions on Evolutionary Computation, 24(1), 69-83.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Tran Dinh Nam**  
Hanoi University of Science and Technology  
📧 Contact: [GitHub](https://github.com/Nemztran)

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful!</strong>
</div>
