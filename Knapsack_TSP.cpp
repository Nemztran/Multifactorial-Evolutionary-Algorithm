#include<bits/stdc++.h>
#include<algorithm>
#include<iostream>
#include<vector>
using namespace std;
const int POP_SIZE = 100;
const int MAX_GENERATIONS = 500;
const double rmp = 0.3 ;
const double lambda = 1e9;
int problemDim;
struct TSP
   {
       vector<vector<double>> distance;
   };
struct Knapsack
   {
       double capacity;
       vector<double> weight;
       vector<double> profit;
   };
struct Problem
{
   TSP tsp;
   Knapsack knapsack;
};
Problem problems;
void init()
{
     problems.tsp.distance.resize(problemDim,vector<double> (problemDim));
     problems.knapsack.weight.resize(problemDim);
     problems.knapsack.profit.resize(problemDim);
     for (int i = 0 ; i < problemDim; i++)
        for (int j = 0 ; j < problemDim; j++)
             cin >> problems.tsp.distance[i][j];
    cin >> problems.knapsack.capacity;
    for (int i = 0 ; i < problemDim; i++)
    {
        cin >> problems.knapsack.weight[i];
        cin >> problems.knapsack.profit[i];
    }

}
vector<double> initGen()
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis (0.0,1.0);
    vector<double> gens(problemDim);
    for (int i = 0 ; i < problemDim; i++)
         gens[i] = dis(gen);
    return gens;
}
vector<int> decode(const string& problem, vector<double> Gen)
{
    vector<int> chromosomes(problemDim);
    if (problem == "Knapsack")
    {
       for (int i = 0 ; i < problemDim; i++)
         if (Gen[i] >= 0.5) chromosomes[i] = 1;
         else chromosomes[i] = 0;
    }
    else
    {
        vector<double> temp = Gen;
        sort(Gen.begin(),Gen.end());
        for (int i = 0 ; i < problemDim; i++)
        {
            auto it = find(Gen.begin(),Gen.end(),temp[i]);
            chromosomes[i] = distance(Gen.begin(),it);
        }
    }
    return chromosomes;
}
struct Individual
{
    vector<double> gen;
    int skillFactor;               // task ma no hoat dong tot nhat
    vector<double> factorialCost;  // chua fitness cua ca the trong tung task
    vector<int> factorialRank;  // vi tri cua ca the trong quan the trong tung task
    double scalarFitness;             // độ phù hợp vô hướng
};
double cal_factorialCost(const vector<double>& Gen,const string& problem, const struct Problem problems)
{
        vector<int> ind = decode(problem,Gen);
        if (problem == "Knapsack") {
             double current_weight = 0, current_profit=0;
             for (int i = 0 ; i < problemDim; i++)
                if (ind[i] == 1) {
                     current_weight += problems.knapsack.weight[i];
                     current_profit += problems.knapsack.profit[i];
                }
        double constraintviolation = max(0.0, current_weight - problems.knapsack.capacity);
       // if (current_weight>problems.knapsack.capacity) constraintviolation = 0;
           return -current_profit + lambda*constraintviolation;
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
    if (it!=fitness.end()) return distance(fitness.begin(),it)+1;
    return -1;
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
        Pop[i].factorialCost[1] = cal_factorialCost(Pop[i].gen,"Knapsack",problems);
    }
    for (int i = 0 ; i < POP_SIZE;i++)
    {
        Pop[i].skillFactor = (Pop[i].factorialCost[0]>abs(Pop[i].factorialCost[1])) ? 0 : 1 ;
        Pop[i].factorialRank[0] = cal_factorialRank(Pop,Pop[i],"TSP");
        Pop[i].factorialRank[1] = cal_factorialRank(Pop,Pop[i],"Knapsack");
        Pop[i].scalarFitness = 1.0 / min(Pop[i].factorialRank[0],Pop[i].factorialRank[1]);
    }
    return Pop;
}
void CrossOver(const Individual& parent1, const Individual& parent2, Individual& child1, Individual& child2)      // Simulated Binary Crossover
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis (0.0,1.0);
    double u,beta,n;
    n = 1;
    u = dis(gen);
    if (u<=0.5) beta = pow(2*u,1.0/(n+1));
    else beta = pow(2*(1-u),-1.0/(n+1));
    for (int i = 0 ; i < problemDim; i++)
        {
            child1.gen[i] = 0.5 * ((1+beta)*parent1.gen[i] + (1-beta)*parent2.gen[i]);
            child2.gen[i] = 0.5 * ((1-beta)*parent1.gen[i] + (1+beta)*parent2.gen[i]);
        }
}
void slightMutation(const Individual& parent, Individual& child)
{
    random_device rd;
    mt19937 gen(rd());
    double biendodotbien = 0.05;
    uniform_real_distribution<> dis (0.0,biendodotbien*biendodotbien);
    double random = dis(gen);
    for (int i = 0 ; i < problemDim; i++)
      child.gen[i] = parent.gen[i] + random;

}
void evaluateSingle(Individual& child, const Individual& parent)
{
    int task = parent.skillFactor;
    string Problem = (task > 0) ? "Knapsack" : "TSP";
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
        intermediate_pop[i].skillFactor = (intermediate_pop[i].factorialCost[0]>abs(intermediate_pop[i].factorialCost[1])) ? 0 : 1 ;
        intermediate_pop[i].factorialRank[0] = cal_factorialRank(intermediate_pop,intermediate_pop[i],"TSP");
        intermediate_pop[i].factorialRank[1] = cal_factorialRank(intermediate_pop,intermediate_pop[i],"Knapsack");
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
     vector<Individual> current_pop = initPopulatation();
     for (int gen = 0 ; gen < MAX_GENERATIONS; gen++)
     {
         vector<Individual> offspring_pop = generateOffspring_pop(current_pop);
         current_pop = update(current_pop,offspring_pop);
       if (gen%10==0)  cout << "Generated completely gen " << gen << endl;
     }
     double ans_TSP = 1e9, ans_Knapsack = 1e9;
     vector<int> bestTour(problemDim);
     vector<int> bestChoice(problemDim);
     for (int i = 0 ; i < POP_SIZE; i++)
     {
        if (current_pop[i].factorialCost[0] < ans_TSP)
           {
              ans_TSP = current_pop[i].factorialCost[0];
              bestTour = decode("TSP",current_pop[i].gen);
           }
        if (current_pop[i].factorialCost[1] < ans_Knapsack)
           {
             ans_Knapsack = current_pop[i].factorialCost[1];
             bestChoice = decode("Knapsack",current_pop[i].gen);
           }
     }
     cout << ans_TSP << " " << abs(ans_Knapsack) <<endl;
     for (int i = 0 ; i < problemDim; i++) cout << bestTour[i] << " ";


}
