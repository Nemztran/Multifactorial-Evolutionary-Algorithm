#include<bits/stdc++.h>
#include "fls_gls_2opt.h"
using namespace std;
const int POP_SIZE = 500;
const int MAX_GENERATIONS = 500;
const double rmp = 0.4;
int problemDim;
struct TSP
    {
       vector<vector<double>> distance;
    };
struct TRP
   {
        vector<vector<double>> time;
   };
struct Problem
{
   TSP tsp;
   TRP trp;
};
Problem problems;
void init()
{
     problems.tsp.distance.resize(problemDim,vector<double> (problemDim));
     problems.trp.time.resize(problemDim,vector<double> (problemDim));
     for (int i = 0 ; i < problemDim; i++)
        for (int j = 0 ; j < problemDim; j++)
             cin >> problems.tsp.distance[i][j];
     for (int i = 0 ; i < problemDim; i++)
        for (int j = 0 ; j < problemDim; j++)
             cin >> problems.trp.time[i][j];

}
vector<int> initGen()
{
    vector<int> tour(problemDim);
    for (int i = 0 ; i < problemDim; i++) tour[i] = i;
    random_device rd;
    mt19937 gen(rd());
    shuffle(tour.begin(),tour.end(),gen);
    return tour;
}
struct Individual
{
    vector<int> gen;
    int skillFactor;               // task ma no hoat dong tot nhat
    vector<double> factorialCost;  // chua fitness cua ca the trong tung task
    vector<int> factorialRank;  // vi tri cua ca the trong quan the trong tung task
    double scalarFitness;             // độ phù hợp vô hướng
};
double cal_factorialCost(const vector<int>& ind,const string& problem, const struct Problem problems)
{
        if (problem == "TRP") {
             double prefix = 0 ;
             double totaltime = 0;
             for (int i = 0; i < problemDim- 1;i++) {
                totaltime += prefix + problems.trp.time[ind[i]][ind[i+1]];
                prefix += problems.trp.time[ind[i]][ind[i+1]];
             }
             return totaltime;
        }
        else {
            double totaldistances = 0;
            for (int i = 0; i < problemDim; i++)
            {
                totaldistances += problems.tsp.distance[ind[i]][ind[(i+1)%problemDim]];
            }
            return totaldistances;
        }
}
int cal_factorialRank ( const vector<Individual> POP,const struct Individual ind, const string& problem)
{
     vector<double> fitness(POP.size()) ;
     int task = (problem == "TSP") ? 0 : 1;
     for (int i = 0 ; i < POP.size(); i++)
         fitness[i] = POP[i].factorialCost[task];
    sort(fitness.begin(),fitness.end());
    auto it = find (fitness.begin(),fitness.end(),ind.factorialCost[task]);
    return distance(fitness.begin(),it)+1;
}
vector<Individual> initPopulatation()
{
    vector<Individual> Pop(POP_SIZE);
    for (int i = 0 ; i < POP_SIZE; i++)
    {
        Pop[i].gen.resize(problemDim);
        Pop[i].gen = initGen();
        Pop[i].factorialCost.resize(2);
        Pop[i].factorialRank.resize(2);
        Pop[i].factorialCost[0] = cal_factorialCost(Pop[i].gen,"TSP",problems);
        Pop[i].factorialCost[1] = cal_factorialCost(Pop[i].gen,"TRP",problems);
    }
    for (int i = 0 ; i < POP_SIZE;i++)
    {
        Pop[i].skillFactor = (Pop[i].factorialCost[0]>abs(Pop[i].factorialCost[1])) ? 0 : 1 ;
        Pop[i].factorialRank[0] = cal_factorialRank(Pop,Pop[i],"TSP");
        Pop[i].factorialRank[1] = cal_factorialRank(Pop,Pop[i],"TRP");
        Pop[i].scalarFitness = 1.0 / min(Pop[i].factorialRank[0],Pop[i].factorialRank[1]);
    }
    return Pop;
}
// PMX Crossover
void CrossOver(const Individual& parent1, const Individual& parent2, Individual& child1, Individual& child2)
{
    vector<int> parent_1 = parent1.gen;
    vector<int> parent_2 = parent2.gen;
    vector<int> child_1(problemDim, -1);
    vector<int> child_2(problemDim, -1);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0, problemDim - 1);
    int cut1 = dist(gen), cut2 = dist(gen);
    while (cut1 == cut2) cut2 = dist(gen);
    if (cut1 > cut2) swap(cut1, cut2);

    unordered_map<int, int> mapping1, mapping2;
    // Copy đoạn giữa và tạo ánh xạ
    for (int i = cut1; i <= cut2; i++) {
        child_1[i] = parent_2[i];
        child_2[i] = parent_1[i];
        mapping1[parent_2[i]] = parent_1[i];
        mapping2[parent_1[i]] = parent_2[i];
    }
    auto pmx_fill = [&](const vector<int>& donor, vector<int>& child, int cut1, int cut2, unordered_map<int, int>& mapping) {
        for (int i = 0; i < donor.size(); ++i) {
            if (i >= cut1 && i <= cut2) continue;
            int gene = donor[i];
            while (find(child.begin() + cut1, child.begin() + cut2 + 1, gene) != child.begin() + cut2 + 1) {
                if (mapping.count(gene)) gene = mapping[gene];
                else break;
            }
            child[i] = gene;
        }
    };
    pmx_fill(parent_1, child_1, cut1, cut2, mapping1);
    pmx_fill(parent_2, child_2, cut1, cut2, mapping2);

    child1.gen = child_1;
    child2.gen = child_2;
}
// Inversion mutation
void slightMutation(const Individual& parent, Individual& child)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution dist(0,problemDim - 1);
    int i = dist(gen), j = dist(gen);
    while (i == j) j = dist(gen);
    if (i > j) swap(i,j);
    vector<int> chromosome = parent.gen;
    reverse(chromosome.begin()+i,chromosome.begin() + j + 1);
    child.gen = chromosome;

}
void evaluateSingle(Individual& child, const Individual& parent)
{
    int task = parent.skillFactor;
    string Problem = (task > 0) ? "TRP" : "TSP";
    child.factorialCost[task] = cal_factorialCost(child.gen,Problem,problems);
    int remaintask = abs(1-task);
    child.factorialCost[remaintask] = std::numeric_limits<double>::infinity();
    child.skillFactor = task;
}
vector<Individual> generateOffspring_pop(vector<Individual>& current_pop)
{
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> distReal(0.0,1.0);
        uniform_int_distribution<> distInt(0,POP_SIZE-1);
        vector<Individual> offspring_pop(POP_SIZE);
        for (int i = 0 ; i < POP_SIZE; i++)
        {
            offspring_pop[i].gen.resize(problemDim);
            offspring_pop[i].factorialCost.resize(2);
            offspring_pop[i].factorialRank.resize(2);
        }
        for (int i = 0 ; i < POP_SIZE/2 ; i++)
       {
           double random = distReal(gen);
           int index_a = distInt(gen);int index_b;
           do {
               index_b = distInt(gen);
           } while (index_a == index_b);
          Individual parent_a = current_pop[index_a];
          Individual parent_b = current_pop[index_b];
          if (parent_a.skillFactor == parent_b.skillFactor||random < rmp)
            {
                 CrossOver(parent_a,parent_b,offspring_pop[2*i],offspring_pop[2*i+1]);
                 int index_1 = distReal(gen), index_2 = distReal(gen);
                 if (index_1<0.5) {
                        evaluateSingle(offspring_pop[2*i],parent_a);
                 }
                 else evaluateSingle(offspring_pop[2*i],parent_b);
                 if (index_2<0.5) {
                    evaluateSingle(offspring_pop[2*i+1],parent_a);
                 }
                 else evaluateSingle(offspring_pop[2*i+1],parent_b);
            }
          else
           {
                slightMutation(parent_a,offspring_pop[2*i]);
                evaluateSingle(offspring_pop[2*i],parent_a);
                slightMutation(parent_b,offspring_pop[2*i+1]);
                evaluateSingle(offspring_pop[2*i+1],parent_b);
           }

       }
    return offspring_pop;
}
bool cmpscalarfitness(const Individual&a, const Individual&b)
{
    return a.scalarFitness > b.scalarFitness;
}
vector<Individual> update (const vector<Individual> current_pop, const vector<Individual> offspring_pop)
{
      vector<Individual> intermediate_pop = current_pop;
      intermediate_pop.insert(intermediate_pop.end(),offspring_pop.begin(),offspring_pop.end());
      for (int i = 0 ; i < 2*POP_SIZE;i++)
    {
        intermediate_pop[i].skillFactor = (intermediate_pop[i].factorialCost[0]>intermediate_pop[i].factorialCost[1]) ? 0 : 1 ;
        intermediate_pop[i].factorialRank[0] = cal_factorialRank(intermediate_pop,intermediate_pop[i],"TSP");
        intermediate_pop[i].factorialRank[1] = cal_factorialRank(intermediate_pop,intermediate_pop[i],"TRP");
        intermediate_pop[i].scalarFitness = 1.0 / min(intermediate_pop[i].factorialRank[0],intermediate_pop[i].factorialRank[1]);
    }
    sort(intermediate_pop.begin(),intermediate_pop.end(),cmpscalarfitness);
    vector<Individual> updated(intermediate_pop.begin(),intermediate_pop.begin()+POP_SIZE);
   return updated;
}

int main ()
{
    cin >> problemDim;
    init();

    double ans_TSP = 1e9, ans_TRP = 1e9;
    vector<int> bestTour(problemDim);
    vector<int> bestChoice(problemDim);
     int GLS_ITER = 100;
    double alpha = 0.3;
    vector<Individual> current_pop = initPopulatation();
 for (int gen = 0; gen < MAX_GENERATIONS; gen++)
{
    vector<Individual> offspring_pop = generateOffspring_pop(current_pop);
    //  Chạy GLS cho cá thể tốt nhất bài TSP vaf TRP
    int bestTSP = 0, bestTRP = 0;
    double minTSP = 1e18, minTRP = 1e18;
    for (int i = 0; i < POP_SIZE; i++) {
        if (current_pop[i].factorialCost[0] < minTSP && current_pop[i].factorialCost[0] < 1e18) {
            minTSP = current_pop[i].factorialCost[0];
            bestTSP = i;
        }
        if (current_pop[i].factorialCost[1] < minTRP && current_pop[i].factorialCost[1] < 1e18) {
            minTRP = current_pop[i].factorialCost[1];
            bestTRP = i;
        }
    }
    // TSP
    vector<int> improvedTSP = runGLS_FLS(current_pop[bestTSP].gen, problems.tsp.distance, GLS_ITER, alpha);
    current_pop[bestTSP].gen = improvedTSP;
    current_pop[bestTSP].factorialCost[0] = cal_factorialCost(improvedTSP, "TSP", problems);
    // TRP
    vector<int> improvedTRP = runGLS_FLS(current_pop[bestTRP].gen, problems.trp.time, GLS_ITER, alpha);
    current_pop[bestTRP].gen = improvedTRP;
    current_pop[bestTRP].factorialCost[1] = cal_factorialCost(improvedTRP, "TRP", problems);
    // tiep tuc update nhu trong MFEA
    current_pop = update(current_pop, offspring_pop);
    if (gen % 10 == 0) cout << "Generated completely gen " << gen << endl;
    if (gen == 10 || gen == 50 || gen == 100 || gen == 150 || gen ==200 || gen == 250 || gen == 300) {
          for (int i = 0 ; i < POP_SIZE; i++)
    {
        if (current_pop[i].factorialCost[0] < ans_TSP)
        {
            ans_TSP = current_pop[i].factorialCost[0];
            bestTour = current_pop[i].gen;
        }
        if (current_pop[i].factorialCost[1] < ans_TRP)
        {
            ans_TRP = current_pop[i].factorialCost[1];
            bestChoice = current_pop[i].gen;
        }
    }
    cout << "GEN" << gen << ": " << ans_TSP << " " << ans_TRP << endl;
    }
}
    cout << "Best TSP tour: ";
    cout << endl;
for (int i = 0; i < problemDim; i++) cout << bestTour[i] << " ";
    cout << ans_TSP << " " << ans_TRP <<endl;
}
