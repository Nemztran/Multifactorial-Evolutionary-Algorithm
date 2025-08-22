#include <iostream>
#include <vector>
#include <random>
#include "fls_gls_2opt.h"
using namespace std;

int main() {
    int n =10; // Số đỉnh
    vector<vector<double>> matrix(n, vector<double>(n));
    // Sinh ngẫu nhiên ma trận chi phí đối xứng, không âm
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1, 100);
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j) {
            matrix[i][j] = matrix[j][i] = dis(gen);
        }
    // Sinh tour ngẫu nhiên
    vector<int> tour(n);
    for (int i = 0; i < n; ++i) tour[i] = i;
    shuffle(tour.begin(), tour.end(), gen);
    cout << "Tour ban dau: ";
    for (int x : tour) cout << x << " ";
    cout << endl;

    // Chạy GLS
    vector<int> result = runGLS_FLS(tour, matrix, 20, 0.1);

    cout << "Tour sau GLS: ";
    for (int x : result) cout << x << " ";
    cout << endl;

    return 0;
}