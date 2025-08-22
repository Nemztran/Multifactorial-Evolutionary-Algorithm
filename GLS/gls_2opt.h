#pragma once
#include <vector>
#include <algorithm>
#include <limits>
using namespace std;

// Tính augmented cost (chi phí có phạt)
 double augmented_cost( const vector<int>& tour,const vector<vector<double>>& matrix, const vector<vector<int>>& penalty, double lambda)
 {
    double cost = 0.0;
    int n = tour.size();
    for (int i = 0; i < n; ++i) {
        int u = tour[i], v = tour[(i + 1) % n];
        cost += matrix[u][v] + lambda * penalty[u][v];
    }
    return cost;
}
 void update_penalty( const vector<int>& tour, const vector<vector<double>>& matrix, vector<vector<int>>& penalty) 
 {
    int n = tour.size();
    double max_util = -1.0;
    vector<pair<int, int>> max_edges;
    for (int i = 0; i < n; ++i) {
        int u = tour[i], v = tour[(i + 1) % n];
        double util = matrix[u][v] / (1.0 + penalty[u][v]);
        if (util > max_util + 1e-9) {
            max_util = util;
            max_edges.clear();
            max_edges.push_back({u, v});
        } else if (abs(util - max_util) < 1e-9) {
            max_edges.push_back({u, v});
        }
    }
    for (auto [u, v] : max_edges) {
        penalty[u][v]++;
        penalty[v][u]++;
    }
}

// Cập nhật penalty cho các cạnh có utility lớn nhất
// Local search 2-opt với augmented cost
 vector<int> opt2_gls( const vector<int>& tour, const vector<vector<double>>& matrix, const vector<vector<int>>& penalty, double lambda) 
 {
    int n = tour.size();
    vector<int> bestTour = tour;
    // Tính augmented cost ban đầu
    double bestCost = 0.0;
    for (int i = 0; i < n; ++i) {
        int u = bestTour[i], v = bestTour[(i + 1) % n];
        bestCost += matrix[u][v] + lambda * penalty[u][v];
    }
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 1; i < n - 2; ++i) {
            for (int j = i + 1; j < n - 1; ++j) {

                int a = bestTour[(i - 1 + n) % n];
                int b = bestTour[i];
                int c = bestTour[j];
                int d = bestTour[(j + 1) % n];

                double old_edges = (matrix[a][b] + lambda * penalty[a][b]) +
                                   (matrix[c][d] + lambda * penalty[c][d]);
                double new_edges = (matrix[a][c] + lambda * penalty[a][c]) +
                                   (matrix[b][d] + lambda * penalty[b][d]);
                double delta = new_edges - old_edges;

                if (delta < -1e-9) {
                    reverse(bestTour.begin() + i, bestTour.begin() + j + 1);
                    bestCost += delta;
                    improved = true;
                }
            }
        }
    }
    return bestTour;
}

// Hàm chính chạy GLS
 vector<int> runGLS( const vector<int>& tour,const vector<vector<double>>& matrix, int max_iter, double lambda) 
{
    int n = tour.size();
    vector<vector<int>> penalty(n, vector<int>(n, 0));
    vector<int> bestTour = tour;
    double bestCost = augmented_cost(tour, matrix, penalty, lambda);
    int no_improve = 0;
    for (int iter = 0; iter < max_iter; ++iter) {
        vector<int> newTour = opt2_gls(bestTour, matrix, penalty, lambda);
        double newCost = augmented_cost(newTour, matrix, penalty, lambda);
        if (newCost < bestCost - 1e-6) {
            bestTour = newTour;
            bestCost = newCost;
        }
        update_penalty(bestTour, matrix, penalty);
    }
    return bestTour;
}