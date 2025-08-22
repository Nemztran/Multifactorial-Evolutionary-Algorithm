#pragma once
#include <vector>
#include <algorithm>
#include <limits>
using namespace std;

double augmented_cost(  const vector<int>& tour,const vector<vector<double>>& matrix, const vector<vector<int>>& penalty, double lambda) 
{
    double cost = 0.0;
    int n = tour.size();
    for (int i = 0; i < n; ++i) {
        int u = tour[i], v = tour[(i + 1) % n];
        cost += matrix[u][v] + lambda * penalty[u][v];
    }
    return cost;
}
vector<int> fls_2opt(  vector<int>& tour,const vector<vector<double>>& matrix,const vector<vector<int>>& penalty, double lambda, vector<int>& activation) 
{
    int n = tour.size();
    vector<int> pos_in_tour(n);
    for (int i = 0; i < n; ++i) pos_in_tour[tour[i]] = i;
    bool improved = false;
    while (any_of(activation.begin(), activation.end(), [](int b){ return b == 1; })) {
        improved = false;
         for (int city = 0; city < n; ++city) {
            if (activation[city] == 1) {
                int i = pos_in_tour[city];
                for (int dir = -1; dir <= 1; dir += 2) {
                    int base1 = i;
                    int base2 = (i + dir + n) % n;
                    if (base1 == base2) continue;
                    int a = tour[base1], b = tour[base2];
                    for (int j = 0; j < n; ++j) {
                        int c = tour[j], d = tour[(j + 1) % n];
                        if (j == base1 || j == base2 || (j + 1) % n == base1 || (j + 1) % n == base2)
                            continue;
                        int left = min(base2, j) + 1;
                        int right = max(base2, j);
                        if (left > right) continue;
                        // Tạo bản sao tour và thử reverse
                        vector<int> new_tour = tour;
                        reverse(new_tour.begin() + left, new_tour.begin() + right + 1);

                        double old_cost = augmented_cost(tour, matrix, penalty, lambda);
                        double new_cost = augmented_cost(new_tour, matrix, penalty, lambda);
                        if (new_cost < old_cost - 1e-9) {
                            tour = new_tour;
                            for (int k = 0; k < n; ++k) pos_in_tour[tour[k]] = k;
                            activation[a] = activation[b] = activation[c] = activation[d] = 1;
                            improved = true;
                            goto ImprovingMoveFound;
                        }
                    }
                }
                activation[city] = 0;        
            }
            ImprovingMoveFound : continue;
        }
        if (!improved) break;
    }
    return tour;
}

// GLS + FLS
  vector<int> runGLS_FLS(const vector<int>& init_tour, const vector<vector<double>>& matrix, int max_iter, double alpha)
{
    int n = init_tour.size();
    vector<vector<int>> penalty(n, vector<int>(n, 0));
    vector<int> bestTour = init_tour;
    double bestCost = augmented_cost(bestTour, matrix, penalty, alpha);

    vector<int> activation(n, 1);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Tính lambda động
        double avg_cost = 0.0;
        for (int i = 0; i < n; ++i) {
            int u = bestTour[i], v = bestTour[(i + 1) % n];
            avg_cost += matrix[u][v];
        }
        avg_cost /= n;
        double lambda = alpha * avg_cost;

        vector<int> newTour = fls_2opt(bestTour, matrix, penalty, lambda, activation);
        double newCost = augmented_cost(newTour, matrix, penalty, lambda);

        if (newCost < bestCost - 1e-6) {
            bestTour = newTour;
            bestCost = newCost;
        }

        // 2. Tính utility cho từng cạnh
        // tăng penalty cho các cạnh có utility lớn nhất
        double max_util = -1.0;
        vector<pair<int, int>> max_edges;
        for (int i = 0; i < n; ++i) {
            int u = bestTour[i], v = bestTour[(i + 1) % n];
            double util = matrix[u][v] / (1.0 + penalty[u][v]);
            if (util > max_util + 1e-9) {
                max_util = util;
                max_edges.clear();
                max_edges.push_back({u, v});
            } else if (abs(util - max_util) < 1e-9) {
                max_edges.push_back({u, v});
            }
        }
        // 3. Tăng penalty và kích hoạt lại các sub-neighborhood liên quan
        for (auto [u, v] : max_edges) {
            penalty[u][v]++;
            penalty[v][u]++;
            activation[u] = activation[v] = 1; // kích hoạt lại các bit liên quan
        }
        if (max_edges.empty()) break;
    }
    return bestTour;
}
