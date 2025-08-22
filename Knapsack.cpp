#include<iostream>
#include<algorithm>
#include<random>
#include<vector>
#include<ctime>
#include<limits>
using namespace std;

#define POP_SIZE 50
#define MAX_GENERATIONS 100
#define rmp 0.2
const int numTask = 5;
int numcities = 10; // Giả sử số thành phố là 10
vector<vector<vector<double>>> distances(numTask);

// Cấu trúc cá thể
struct Individual {
    vector<int> tour;
    int skillFactor;
    double fitness;
    vector<double> taskFitness;
};

// Các hàm đã định nghĩa trước đó
vector<vector<double>> init() {
    vector<vector<double>> distance(numcities, vector<double>(numcities));
    for (int i = 0; i < numcities; i++)
        for (int j = 0; j < numcities; j++)
            cin >> distance[i][j];
    return distance;
}

vector<vector<double>> generatedistance() {
    vector<vector<double>> distance(numcities, vector<double>(numcities));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 1000.0);
    for (int i = 0; i < numcities; i++)
        for (int j = 0; j < numcities; j++)
            if (j != i) distance[i][j] = dis(gen);
            else distance[i][j] = 0;
    return distance;
}

vector<Individual> initializepopulation() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> disTask(0, numTask - 1);
    vector<Individual> population(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) {
        population[i].tour.resize(numcities);
        for (int j = 0; j < numcities; j++)
            population[i].tour[j] = j;
        shuffle(population[i].tour.begin(), population[i].tour.end(), gen);
        population[i].skillFactor = disTask(gen);
        population[i].fitness = 0;
        population[i].taskFitness.resize(numTask);
    }
    return population;
}

double tourdistance(const vector<int>& tour, const vector<vector<double>>& distance) {
    double s = distance[tour[numcities - 1]][tour[0]];
    for (int i = 0; i < tour.size() - 1; i++)
        s += distance[tour[i]][tour[i + 1]];
    return s;
}

void evaluateIndividual(Individual& ind) {
    for (int task = 0; task < numTask; task++)
        ind.taskFitness[task] = tourdistance(ind.tour, distances[task]);
}

void evaluateIndividualSingle(Individual& ind, int task) {
    fill(ind.taskFitness.begin(), ind.taskFitness.end(), numeric_limits<double>::infinity());
    ind.taskFitness[task] = tourdistance(ind.tour, distances[task]);
}

void skillFactor(Individual& ind) {
    auto minIt = min_element(ind.taskFitness.begin(), ind.taskFitness.end());
    if (*minIt == numeric_limits<double>::infinity()) {
        ind.skillFactor = -1; // Không hợp lệ
    } else {
        ind.skillFactor = distance(ind.taskFitness.begin(), minIt); // Chỉ số task tốt nhất
    }
}

// Hàm orderCrossover
void orderCrossover(Individual parent1, Individual parent2, Individual& child1, Individual& child2) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, numcities - 1);
    int start = dis(gen);
    int end = dis(gen);
    if (start > end) swap(start, end);

    vector<bool> used1(numcities, false);
    vector<bool> used2(numcities, false);

    // Sao chép đoạn từ parent1 vào child1 và từ parent2 vào child2
    for (int i = start; i <= end; i++) {
        child1.tour[i] = parent1.tour[i];
        used1[child1.tour[i]] = true;
        child2.tour[i] = parent2.tour[i];
        used2[child2.tour[i]] = true;
    }

    // Điền các thành phố còn lại
    int curpos1 = (end + 1) % numcities;
    int curpos2 = (end + 1) % numcities;
    for (int i = 0; i < numcities; i++) {
        int idx = (end + 1 + i) % numcities;
        int city1 = parent2.tour[idx];
        int city2 = parent1.tour[idx];
        if (!used1[city1]) {
            child1.tour[curpos1] = city1;
            curpos1 = (curpos1 + 1) % numcities;
        }
        if (!used2[city2]) {
            child2.tour[curpos2] = city2;
            curpos2 = (curpos2 + 1) % numcities;
        }
    }

    // Gán skill factor
    child1.skillFactor = parent1.skillFactor;
    child2.skillFactor = parent2.skillFactor;
}

int main() {
    // Khởi tạo ma trận khoảng cách cho các task
    for (int i = 0; i < numTask; i++) {
        distances[i] = generatedistance();
    }

    // Khởi tạo quần thể
    vector<Individual> population = initializepopulation();

    // Đánh giá quần thể trên tất cả các task
    for (Individual& ind : population) {
        evaluateIndividual(ind);
    }

    // Xác định skill factor cho từng cá thể
    for (Individual& ind : population) {
        skillFactor(ind);
    }

    // Tạo quần thể con bằng cách lai ghép
    vector<Individual> offspring(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i += 2) {
        orderCrossover(population[i], population[i + 1], offspring[i], offspring[i + 1]);
    }

    // In kết quả
    cout << "Parent Population:\n";
    for (int i = 0; i < POP_SIZE; i++) {
        cout << "Individual " << i << ":\n";
        cout << "  Tour: ";
        for (int city : population[i].tour) {
            cout << city << " ";
        }
        cout << "\n  Skill Factor: " << population[i].skillFactor << "\n";
        cout << "  Task Fitness: ";
        for (double fitness : population[i].taskFitness) {
            cout << fitness << " ";
        }
        cout << "\n";
    }

    cout << "\nOffspring Population:\n";
    for (int i = 0; i < POP_SIZE; i++) {
        cout << "Individual " << i << ":\n";
        cout << "  Tour: ";
        for (int city : offspring[i].tour) {
            cout << city << " ";
        }
        cout << "\n  Skill Factor: " << offspring[i].skillFactor << "\n";
        cout << "  Task Fitness: ";
        for (double fitness : offspring[i].taskFitness) {
            cout << fitness << " ";
        }
        cout << "\n";
    }

    return 0;
}
