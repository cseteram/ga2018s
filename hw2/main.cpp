#include <cstdio>
#include <cstdlib>
#include <climits>
#include <ctime>
#include <vector>
#include <algorithm>
#include <tuple>

using namespace std;
typedef pair<int, int> pii;
typedef vector<int> chromosome;

int num_local_search = 0;

// Helper Functions
double GetTime();
void PrintStats(vector<int> &scores);
int Evaluate(const vector<vector<pii>> &graph, chromosome &sol);

// GA architecture
class GA {
private:
    const double ga_duration;
    const int n_populations, k_new_populations;
    const double k_selection_pressure;
    const double p0_uniform_crossover;
    const double p0_mutation;
    
    const int v;
    const vector<vector<pii>> graph;
    const vector<chromosome> initial_populations;

    int generation;
    vector<chromosome> populations;
    vector<int> scores;

    void Generate();
    int Select();
    chromosome Crossover(chromosome &p1, chromosome &p2);
    void Mutation(chromosome &p);
    void Replace(vector<chromosome> &new_populations);

public:
    bool print_status = false;

    GA(double, int, int, double, double, double, vector<vector<pii>>, vector<chromosome>);
    chromosome Run();
};

// Functions for local search
void LocalSearch(const vector<vector<pii>> &graph, chromosome &x);

// Functions for multi-start local search
chromosome MultiStartLocalSearch(int n);

// Functions for main
void Input(char *input, vector<vector<pii>> &graph);
void Output(char *output);

double GetTime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

void PrintStats(vector<int> &scores)
{
    vector<int> scores_copy(scores);
    sort(scores_copy.begin(), scores_copy.end());

    int n = (int)scores_copy.size();
    printf("Top 10 Score =>");
    for (int i = n - 1; i >= n - 10; --i)
        printf(" %d", scores_copy[i]);
    
    int sum = 0;
    for (int i = 0; i < n; ++i)
        sum += scores_copy[i];
    printf("\nmax = %d | min = %d | avg = %.3f\n",
        scores_copy[n - 1],
        scores_copy[0],
        sum / (double)n
    );
}

int Evaluate(const vector<vector<pii>> &graph, chromosome &sol)
{
    int ret = 0;
    for (int here = 0; here < (int)sol.size(); ++here) {
        if (sol[here]) continue;
        for (auto &p : graph[here]) {
            int there, w;
            tie(there, w) = p;
            
            if (!sol[there]) continue;
            ret += w;
        }
    }
    return ret;
}

GA::GA(double du, int n, int k, double kp, double p0c, double p0m, vector<vector<pii>> g, vector<chromosome> x)
    : ga_duration(du), n_populations(n), k_new_populations(k), k_selection_pressure(kp),
      p0_uniform_crossover(p0c), p0_mutation(p0m), v((int)g.size()), graph(g), initial_populations(x)
{

}

void GA::Generate()
{
    int n = n_populations, n0 = (int)initial_populations.size();
    populations.resize(n); scores.resize(n);

    for (int i = 0; i < n0; ++i)
        populations[i] = initial_populations[i];
    for (int i = n0; i < n; ++i) {
        chromosome p(v);
        for (int j = 0; j < v; ++j) {
            p[j] = rand() % 2;

            // int x = rand() % v;
            // if (x < j) p[j] = 0;
            // else p[j] = 1;
        }
        populations[i] = p;
    }

    for (int i = 0; i < n; ++i)
        scores[i] = Evaluate(graph, populations[i]);
}

int GA::Select()
{
    // Roulette-Wheel Selection
    int min_score = *min_element(scores.begin(), scores.end());
    int max_score = *max_element(scores.begin(), scores.end());

    vector<double> fitnesses(n_populations);
    double sum_fitnesses = 0.0;
    for (int i = 0; i < n_populations; ++i) {
        fitnesses[i] = (scores[i] - min_score);
        fitnesses[i] += (max_score - min_score) / (k_selection_pressure - 1);
        sum_fitnesses += fitnesses[i];
    }

    double point = (double)rand() / (double)(RAND_MAX/sum_fitnesses);
    double sum = 0.0;
    for (int i = 0; i < n_populations; i++) {
        sum += fitnesses[i];
        if (point < sum) return i;
    }
    return 0; // will not happen
}

chromosome GA::Crossover(chromosome &px, chromosome &py)
{
    chromosome pz(v);

    // uniform crossover
    for (int i = 0; i < v; ++i) {
        if (px[i] == py[i])
            pz[i] = px[i];
        else {
            int prob = rand() % 1000;
            pz[i] = (prob >= p0_uniform_crossover * 1000) ? px[i] : py[i];
        }
    }

    return pz;
}

void GA::Mutation(chromosome &p)
{
    for (int i = 0; i < v; ++i) {
        int prob = rand() % 1000;
        if (prob < p0_mutation * 1000) {
            p[i] ^= 1;
        }
    }
}

void GA::Replace(vector<chromosome> &new_populations)
{
    // choose only good solutions
    for (auto &np : new_populations) {
        int min_v = INT_MAX, min_id = -1;
        for (int i = 0; i < n_populations; ++i) {
            if (scores[i] < min_v) {
                min_v = scores[i];
                min_id = i;
            }
        }

        int score_np = Evaluate(graph, np);
        if (score_np < min_v) continue;

        populations[min_id] = np;
        scores[min_id] = score_np;
    }
}

chromosome GA::Run()
{
    srand(time(NULL));
    double start_time = GetTime();
    generation = 0;

    Generate();
    do {
        vector<chromosome> new_populations(k_new_populations);
        for (int i = 0; i < k_new_populations; ++i) {
            int x = Select();
            int y = Select();

            auto &px = populations[x], &py = populations[y];
            auto pz = Crossover(px, py);
            Mutation(pz);

            new_populations[i] = pz;
        }

        for (int i = 0; i < k_new_populations; ++i)
            LocalSearch(graph, new_populations[i]);
        
        Replace(new_populations);
        
        ++generation;
        if (print_status) {
            printf("=== Generation: %d ===\n", generation);
            PrintStats(scores);
        }

        double current_time = GetTime();
        if (current_time - start_time >= ga_duration)
            break;
    } while(true);

    auto iter = max_element(scores.begin(), scores.end());
    auto i = distance(scores.begin(), iter);
    return populations[i];
    printf("\nLast Score => %d\n", scores[i]);
}

void LocalSearch(const vector<vector<pii>> &graph, chromosome &sol)
{
    int n = (int)sol.size();
    vector<int> sigma(n);
    for (int i = 0; i < n; ++i)
        sigma[i] = i;

    // generate a random permutation sigma of {1, ..., |V|}
    random_shuffle(sigma.begin(), sigma.end());
    bool improved = true;

    while (improved) {
        improved = false;
        for (int i = 0; i < n; ++i) {
            int here = sigma[i], delta = 0;
            for (auto &p : graph[here]) {
                int there, w;
                tie(there, w) = p;

                if (sol[here] == sol[there])
                    delta += w;
                else
                    delta -= w;
            }
            
            if (delta > 0) {
                sol[here] ^= 1;
                improved = true;
            }
        }
    }

    ++num_local_search;
}

chromosome MultiStartLocalSearch(vector<vector<pii>> &graph, int n)
{
    int v = (int)graph.size();
    chromosome ans;
    int max_score = INT_MIN;

    srand(time(NULL));
    for (int tc = 0; tc < n; ++tc) {
        chromosome x(v);
        for (int i = 0; i < v; ++i)
            x[i] = rand() % 2;
        LocalSearch(graph, x);
        int score = Evaluate(graph, x);
        if (score > max_score) {
            ans = x;
            max_score = score;
        }
    }

    return ans;
}

void Input(char *input, vector<vector<pii>> &graph)
{
    FILE *in = fopen(input, "r");
    if (in == NULL) {
        printf("Can't open the input file/\n");
        exit(EXIT_FAILURE);
    }

    int v, e;
    fscanf(in, "%d%d", &v, &e);
    graph.resize(v);
    for (int i = 0; i < e; ++i) {
        int x, y, w;
        fscanf(in, "%d%d%d", &x, &y, &w);
        --x, --y;
        graph[x].push_back({y, w});
        graph[y].push_back({x, w});
    }
}

void Output(char *output, chromosome &x)
{
    FILE *out = fopen(output, "w");
    if (out == NULL) {
        printf("Can't open the output file.\n");
        exit (EXIT_FAILURE);
    }

    bool first = true;
    for (int i = 0; i < (int)x.size(); ++i) {
        if (x[i]) continue;
        if (!first) fprintf(out, " ");
        first = false;
        fprintf(out, "%d", i + 1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s [input_file] [output_file]\n", argv[0]);
        printf("Example : %s maxcut.in maxcut.out\n", argv[0]);
        return EXIT_FAILURE;
    }

    vector<vector<pii>> graph;
    Input(argv[1], graph);

    vector<chromosome> v;

    GA layer1(1.0, 256, 32, 4, 0.5, 0.01, graph, v);
    layer1.print_status = true;
    for (int i = 0; i < 64; ++i) {
        chromosome x = layer1.Run();
        v.push_back(x);
    }

    GA layer2(3.0, 512, 32, 4, 0.5, 0.01, graph, v);
    layer2.print_status = true;
    for (int i = 0; i < 16; ++i) {
        chromosome x = layer2.Run();
        v.push_back(x);
    }

    GA layer3(20.0, 1024, 64, 4, 0.5, 0.01, graph, v);
    layer3.print_status = true;
    for (int i = 0; i < 3; ++i) {
        chromosome x = layer3.Run();
        v.push_back(x);
    }

    GA layer4(3.0, 1024, 64, 4, 0.5, 0.01, graph, v);
    layer4.print_status = true;
    chromosome x = layer4.Run();

    Output(argv[2], x);

    // printf("Local Search = %d\n", num_local_search);
    // chromosome x = MultiStartLocalSearch(graph, 2686490);
    // printf("Score = %d\n", Evaluate(graph, x));

    return 0;
}
