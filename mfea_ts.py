import numpy as np
from copy import deepcopy
import random


class TSPProblem:
    def __init__(self, distance_matrix):
        self.distance_matrix = np.array(distance_matrix)
        self.n = len(distance_matrix)

    def calculate_cost(self, tour):
        total_cost = 0
        for i in range(len(tour) - 1):
            total_cost += self.distance_matrix[tour[i]][tour[i + 1]]
        total_cost += self.distance_matrix[tour[-1]][tour[0]]
        return total_cost if total_cost > 0 else 1e-10

class TRPPDProblem:
    def __init__(self, travel_time_matrix, debris_time_matrix, priorities, block_edges=None):
        self.travel_time_matrix = np.array(travel_time_matrix)
        self.debris_time_matrix = np.array(debris_time_matrix)
        self.priorities = np.array(priorities)
        self.n = len(travel_time_matrix)
        self.block_edges = block_edges 

    def calculate_cost(self, tour):
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            if (u, v) in self.block_edges or (v, u) in self.block_edges:
                return float('inf')
        total_cost = 0
        current_time = 0
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            current_time += self.travel_time_matrix[u][v] + self.debris_time_matrix[u][v]
            if v != 0:
                total_cost += current_time * self.priorities[v]
        current_time += self.travel_time_matrix[tour[-1]][tour[0]] + self.debris_time_matrix[tour[-1]][tour[0]]
        return total_cost
class Individual:
    def __init__(self, num_task, n):
        self.chromosome = None
        self.fitness = [1e9 for _ in range(num_task)]
        self.factorial_rank = np.zeros(num_task)
        self.scalar_fitness = None
        self.skill_factor = None
        self.diversity_score = 0
        self.diversity_rank = 0
        self.n = n

    def gen_init_insertion(self, problem):
        tour = [0]
        unvisited = list(range(1, self.n))
        np.random.shuffle(unvisited)
        while unvisited:
            best_pos = -1
            best_node = -1
            min_insertion_cost = float('inf')
            for node in unvisited:
                for pos in range(1, len(tour) + 1):
                    cost = problem.insertion_cost(tour, pos, node)
                    if cost < min_insertion_cost:
                        min_insertion_cost = cost
                        best_pos = pos
                        best_node = node
            if best_pos == -1:
                break
            tour.insert(best_pos, best_node)
            unvisited.remove(best_node)
        tour.append(0)
        self.chromosome = np.array(tour)

    def repair_algorithm_4(self, problem):
        tour = self.chromosome.tolist()
        seen = {0}
        unique_tour = [0]
        for node in tour[1:-1]:
            if node not in seen and 1 <= node < self.n:
                unique_tour.append(node)
                seen.add(node)
        missing = [x for x in range(1, self.n) if x not in seen]
        for node in missing:
            best_pos = -1
            min_insertion_cost = float('inf')
            for pos in range(1, len(unique_tour) + 1):
                cost = problem.insertion_cost(unique_tour, pos, node)
                if cost < min_insertion_cost:
                    min_insertion_cost = cost
                    best_pos = pos
            if best_pos != -1:
                unique_tour.insert(best_pos, node)
        unique_tour.append(0)
        tour = np.array(unique_tour)

        neighborhoods = [self.remove_insert, self.move_up, self.move_down, self.shift, self.swap_adjacent, self.exchange, self.two_opt, self.or_opt]
        improved = True
        while improved:
            improved = False
            best_tour = tour
            best_cost = problem.calculate_cost(tour)
            for nh in neighborhoods:
                for i in range(1, len(tour) - 1):
                    for j in range(1, len(tour) - 1):
                        if i == j:
                            continue
                        neighbor = nh(tour, i, j)
                        if neighbor is not None and self.is_feasible(neighbor, problem):
                            cost = problem.calculate_cost(neighbor)
                            if cost < best_cost:
                                best_tour = neighbor
                                best_cost = cost
                                improved = True
            tour = best_tour
        self.chromosome = tour

    def is_feasible(self, tour, problem):
        if len(tour) != self.n + 1 or tour[0] != 0 or tour[-1] != 0 or len(set(tour[1:-1])) != self.n - 1:
            return False
        if hasattr(problem, 'block_edges'):
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i + 1]
                if (u, v) in problem.block_edges or (v, u) in problem.block_edges:
                    return False
        return all(1 <= node < self.n for node in tour[1:-1])

    # Neighborhood functions
    def remove_insert(self, tour, i, j):
        neighbor = tour.copy()
        node = neighbor[i]
        neighbor = np.delete(neighbor, i)
        insert_j = j if j < i else j - 1
        neighbor = np.insert(neighbor, insert_j, node)
        return neighbor

    def move_up(self, tour, i, j):
        if i <= 1 or j >= i or j < 1:
            return None
        neighbor = tour.copy()
        node = neighbor[i]
        neighbor = np.delete(neighbor, i)
        neighbor = np.insert(neighbor, j, node)
        return neighbor

    def move_down(self, tour, i, j):
        if i >= len(tour) - 2 or j <= i or j >= len(tour) - 1:
            return None
        neighbor = tour.copy()
        node = neighbor[i]
        neighbor = np.delete(neighbor, i)
        neighbor = np.insert(neighbor, j, node)
        return neighbor

    def shift(self, tour, i, j):
        if i >= len(tour) - 2 or j >= len(tour) - 1 or j < 1:
            return None
        neighbor = tour.copy()
        segment = neighbor[i:i+2] if i + 2 <= len(tour) - 1 else neighbor[i:i+1]
        neighbor = np.delete(neighbor, slice(i, i + len(segment)))
        insert_j = j if j < i else j - len(segment)
        if insert_j < 1:
            return None
        neighbor = np.insert(neighbor, insert_j, segment)
        return neighbor

    def exchange(self, tour, i, j):
        neighbor = tour.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def two_opt(self, tour, i, j):
        if j <= i:
            return None
        neighbor = tour.copy()
        neighbor[i:j+1] = neighbor[i:j+1][::-1]
        return neighbor

    def swap_adjacent(self, tour, i, j):
        if j != i + 1:
            return None
        neighbor = tour.copy()
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def or_opt(self, tour, i, j):
        neighbor = tour.copy()
        if i + 2 > len(neighbor) - 1:
            return None
        segment = neighbor[i:i+2]
        neighbor = np.delete(neighbor, slice(i, i+2))
        insert_j = j if j < i else j - 2
        neighbor = np.insert(neighbor, insert_j, segment)
        return neighbor

    def tabu_search(self, problem, cal_metric, max_iter=10):
        current = self.chromosome.copy()
        best = current.copy()
        best_cost = cal_metric(best, problem)
        tabu_list = []
        neighborhoods = [self.remove_insert, self.move_up, self.move_down, self.shift, self.swap_adjacent, self.exchange, self.two_opt, self.or_opt]
        for _ in range(max_iter):
            best_neighbor = None
            best_neighbor_cost = float('inf')
            for i in range(1, len(current) - 1):
                for j in range(1, len(current) - 1):
                    if i == j:
                        continue
                    for nh in neighborhoods:
                        neighbor = nh(current, i, j)
                        if neighbor is not None and self.is_feasible(neighbor, problem):
                            neighbor_tuple = tuple(neighbor)
                            if neighbor_tuple not in tabu_list:
                                cost = cal_metric(neighbor, problem)
                                if cost < best_neighbor_cost:
                                    best_neighbor = neighbor
                                    best_neighbor_cost = cost
            if best_neighbor is not None:
                current = best_neighbor
                tabu_list.append(tuple(current))
                if len(tabu_list) > 10:
                    tabu_list.pop(0)
                if best_neighbor_cost < best_cost:
                    best = best_neighbor
                    best_cost = best_neighbor_cost
        self.chromosome = best
        self.fitness[self.skill_factor] = best_cost

    def cal_fitness_task(self, cal_metric, problem, task):
        self.fitness[task] = cal_metric(self.chromosome, problem)

# Define the Population class
class Population:
    def __init__(self, pop_size, problem_list, n):
        self.pop_size = pop_size
        self.indi_list = []
        self.problem_list = problem_list
        self.n = n

    def gen_pop(self, n, cal_metric_list):
        for _ in range(self.pop_size):
            indi = Individual(len(self.problem_list), n)
            task = random.randint(0, len(self.problem_list) - 1)
            indi.gen_init_insertion(self.problem_list[task])
            if not indi.is_feasible(indi.chromosome, self.problem_list[task]):
                indi.gen_init_insertion(self.problem_list[(task + 1) % len(self.problem_list)])
            for t in range(len(self.problem_list)):
                indi.cal_fitness_task(cal_metric_list[t], self.problem_list[t], t)
            self.indi_list.append(indi)

    def call_factorial_rank(self):
        for task in range(len(self.problem_list)):
            self.indi_list.sort(key=lambda x: x.fitness[task])
            for idx, indi in enumerate(self.indi_list):
                indi.factorial_rank[task] = idx + 1

    def call_skill_factor(self):
        for indi in self.indi_list:
            indi.skill_factor = np.argmin(indi.factorial_rank)

    def call_scalar_fitness(self):
        for indi in self.indi_list:
            indi.scalar_fitness = 1 / min(indi.factorial_rank)

    def calculate_diversity_rank(self):
        for indi in self.indi_list:
            total_distance = 0
            for other in self.indi_list:
                if indi != other:
                    distance = np.sum(indi.chromosome[1:-1] != other.chromosome[1:-1])
                    total_distance += distance
            indi.diversity_score = total_distance / (len(self.indi_list) - 1) if len(self.indi_list) > 1 else 0
        sorted_by_diversity = sorted(self.indi_list, key=lambda x: x.diversity_score, reverse=True)
        for rank, indi in enumerate(sorted_by_diversity):
            indi.diversity_rank = rank + 1

    def apply_tabu_search(self, cal_metric_list):
        best_per_task = [sorted(self.indi_list, key=lambda x: x.fitness[t])[0] for t in range(len(self.problem_list))]
        for indi in best_per_task:
            indi.tabu_search(self.problem_list[indi.skill_factor], cal_metric_list[indi.skill_factor])

# Selection operator with ranking function R(T)
def select_parents(pop, NG, alpha):
    candidates = np.random.choice(pop.indi_list, NG, replace=False)
    for indi in candidates:
        RF = indi.factorial_rank[indi.skill_factor]
        RD = indi.diversity_rank
        SP = len(pop.indi_list)
        R = alpha * (SP - RF + 1) + (1 - alpha) * (SP - RD + 1)
        indi.R_score = R
    sorted_by_R = sorted(candidates, key=lambda x: x.R_score, reverse=True)
    return sorted_by_R[0], sorted_by_R[1]

# Crossover operators
def pmx_crossover(parent1, parent2):
    size = len(parent1.chromosome) - 2
    if size < 2:
        return parent1.chromosome.copy(), parent2.chromosome.copy()
    p1, p2 = parent1.chromosome[1:-1], parent2.chromosome[1:-1]
    off1, off2 = np.full(size, -1), np.full(size, -1)
    start, end = sorted([random.randint(0, size-1) for _ in range(2)] if size > 1 else [0, 0])
    off1[start:end] = p1[start:end]
    off2[start:end] = p2[start:end]
    mapping1, mapping2 = {}, {}
    for i in range(start, end):
        mapping1[p2[i]] = p1[i]
        mapping2[p1[i]] = p2[i]
    for i in range(size):
        if off1[i] == -1:
            val = p2[i]
            while val in mapping1:
                val = mapping1[val]
            off1[i] = val
        if off2[i] == -1:
            val = p1[i]
            while val in mapping2:
                val = mapping2[val]
            off2[i] = val
    off1 = np.concatenate(([0], off1, [0]))
    off2 = np.concatenate(([0], off2, [0]))
    return off1, off2

def cx_crossover(parent1, parent2):
    size = len(parent1.chromosome) - 2
    if size < 2:
        return parent1.chromosome.copy(), parent2.chromosome.copy()
    p1, p2 = parent1.chromosome[1:-1], parent2.chromosome[1:-1]
    off1, off2 = np.full(size, -1), np.full(size, -1)
    pos = 0
    while pos < size and np.where(p2 == p1[pos])[0].size > 0:
        cycle = []
        while off1[pos] == -1:
            cycle.append(pos)
            off1[pos] = p1[pos]
            pos_idx = np.where(p2 == p1[pos])[0]
            if pos_idx.size > 0:
                pos = pos_idx[0]
            else:
                break
        for i in cycle:
            if off1[i] == -1:
                off1[i] = p2[i]
        pos = 0
    for i in range(size):
        if off1[i] == -1:
            off1[i] = p2[i]
    pos = 0
    while pos < size and np.where(p1 == p2[pos])[0].size > 0:
        cycle = []
        while off2[pos] == -1:
            cycle.append(pos)
            off2[pos] = p2[pos]
            pos_idx = np.where(p1 == p2[pos])[0]
            if pos_idx.size > 0:
                pos = pos_idx[0]
            else:
                break
        for i in cycle:
            if off2[i] == -1:
                off2[i] = p1[i]
        pos = 0
    for i in range(size):
        if off2[i] == -1:
            off2[i] = p1[i]
    off1 = np.concatenate(([0], off1, [0]))
    off2 = np.concatenate(([0], off2, [0]))
    return off1, off2

def sc_crossover(parent1, parent2):
    size = len(parent1.chromosome) - 2
    if size < 2:
        return parent1.chromosome.copy(), parent2.chromosome.copy()
    p1, p2 = parent1.chromosome[1:-1], parent2.chromosome[1:-1]
    off1, off2 = np.array(p1.copy()), np.array(p2.copy())
    start = random.randint(0, size - 1)
    seq = np.concatenate((p2[start:], p2[:start]))
    used = set(off1)
    off1_new = [x for x in off1 if x in used]
    for node in seq:
        if node not in used:
            off1_new.append(node)
            used.add(node)
    off1 = np.array(off1_new[:size])
    used = set(off2)
    seq = np.concatenate((p1[start:], p1[:start]))
    off2_new = [x for x in off2 if x in used]
    for node in seq:
        if node not in used:
            off2_new.append(node)
            used.add(node)
    off2 = np.array(off2_new[:size])
    off1 = np.concatenate(([0], off1, [0]))
    off2 = np.concatenate(([0], off2, [0]))
    return off1, off2

def crossover(problem, parent1, parent2, cal_metric):
    crossover_types = [pmx_crossover, cx_crossover, sc_crossover]
    off1, off2 = Individual(len(parent1.fitness), parent1.n), Individual(len(parent2.fitness), parent2.n)
    chosen_crossover = random.choice(crossover_types)
    off1.chromosome, off2.chromosome = chosen_crossover(parent1, parent2)
    off1.repair_algorithm_4(problem)
    off2.repair_algorithm_4(problem)
    return off1, off2

# Mutation operator
def mutation(problem, parent):
    off = Individual(len(parent.fitness), parent.n)
    off.chromosome = parent.chromosome.copy()
    if len(off.chromosome) > 3:
        i, j = np.random.choice(range(1, len(parent.chromosome) - 1), 2, replace=False)
        off.chromosome[i], off.chromosome[j] = off.chromosome[j], off.chromosome[i]
    off.repair_algorithm_4(problem)
    return off

# Update rmp (Algorithm 7)
def update_rmp(rmp, best_task, prev_best_task, L, delta=0.05, theta=5, rmp_min=0.1, rmp_max=0.9):
    improved = any(prev_best_task[i] > best_task[i] for i in range(len(best_task)))
    if improved:
        if len(L) >= theta:
            L.pop(random.randint(0, len(L) - 1))
        L.append(rmp)
    else:
        if L:
            rmp = L[random.randint(0, len(L) - 1)]
        rmp += delta * np.random.normal(0, 1)
        rmp = max(min(rmp, rmp_max), rmp_min)
    return rmp

# Elitism selection
def elitism_selection(pop, pop_size):
    pop.indi_list.sort(key=lambda x: x.scalar_fitness, reverse=True)
    elite_size = max(1, int(0.15 * pop_size))
    elite = pop.indi_list[:elite_size]
    remaining_size = pop_size - elite_size
    remaining = np.random.choice(pop.indi_list[elite_size:], remaining_size, replace=False)
    pop.indi_list = list(elite) + list(remaining)

# Main MFEA-TS algorithm
def mfea_ts(problem_list, num_gen, pop_size, rmp, n, NG=3, alpha=0.5):
    cal_metric_list = [lambda x, p: p.calculate_cost(x) for p in problem_list]
    pop = Population(pop_size, problem_list, n)
    pop.gen_pop(n, cal_metric_list)
    pop.call_factorial_rank()
    pop.call_scalar_fitness()
    pop.call_skill_factor()
    pop.calculate_diversity_rank()
    history = []
    best_tours = [None] * len(problem_list)
    best_task = np.array([float('inf')] * len(problem_list))
    prev_best_task = best_task.copy()
    L = []
    for task in range(len(problem_list)):
        for indi in pop.indi_list:
            if indi.fitness[task] < best_task[task]:
                best_task[task] = indi.fitness[task]
                best_tours[task] = indi.chromosome.copy()
    history.append(best_task.copy())
    print("Generation 0:", best_task)

    for gen in range(num_gen):
        off_list = []
        while len(off_list) < pop_size:
            p1, p2 = select_parents(pop, NG, alpha)
            if p1.skill_factor == p2.skill_factor or np.random.rand() < rmp:
                c1, c2 = crossover(problem_list[p1.skill_factor], p1, p2, cal_metric_list[p1.skill_factor])
            else:
                c1 = mutation(problem_list[p1.skill_factor], p1)
                c2 = mutation(problem_list[p2.skill_factor], p2)
            for t in range(len(problem_list)):
                c1.cal_fitness_task(cal_metric_list[t], problem_list[t], t)
                c2.cal_fitness_task(cal_metric_list[t], problem_list[t], t)
            off_list.extend([c1, c2])

        pop.indi_list.extend(off_list[:pop_size - len(off_list)])
        pop.apply_tabu_search(cal_metric_list)
        pop.call_factorial_rank()
        pop.call_scalar_fitness()
        pop.call_skill_factor()
        elitism_selection(pop, pop_size)
        pop.calculate_diversity_rank()

        for task in range(len(problem_list)):
            min_fitness = min(indi.fitness[task] for indi in pop.indi_list)
            best_task[task] = min_fitness
            for indi in pop.indi_list:
                if indi.fitness[task] == min_fitness:
                    if best_tours[task] is None or min_fitness < cal_metric_list[task](best_tours[task], problem_list[task]):
                        best_tours[task] = indi.chromosome.copy()
                    break
        history.append(best_task.copy())
        print(f"Generation {gen + 1}:", best_task)

        rmp = update_rmp(rmp, best_task, prev_best_task, L)
        prev_best_task = best_task.copy()

    for task in range(len(problem_list)):
        print(f"Best tour for Task {task}:", best_tours[task])
    return history

# Example usage with blocked edges for TRPPD
if __name__ == "__main__":
    n = 10
    distance_matrix = np.random.rand(n, n) 
    travel_time_matrix = np.random.rand(n, n) 
    debris_time_matrix = np.random.rand(n, n) 
    priorities = np.random.rand(n)
    block_edges = set()
    np.fill_diagonal(distance_matrix, 0)
    np.fill_diagonal(travel_time_matrix, 0)
    np.fill_diagonal(debris_time_matrix, 0)
    tsp = TSPProblem(distance_matrix)
    trppd = TRPPDProblem(travel_time_matrix, debris_time_matrix, priorities, block_edges)
    problem_list = [tsp, trppd]
    history = mfea_ts(problem_list, num_gen=10, pop_size=10, rmp=0.3, n=n)